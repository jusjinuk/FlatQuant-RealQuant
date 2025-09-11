import os
import time
import gc
import functools
from contextlib import nullcontext

import torch
import torch.nn as nn
import transformers

from flatquant.function_utils import set_require_grad_all, get_n_set_parameters_byname, get_paras_dict_by_name, check_params_grad
from flatquant.quant_utils import set_quantizer_state


def trainable_parameters_num(model, name = None):
    params = []
    total = 0
    for n, m in model.named_parameters():
        if name is not None:
            if m.requires_grad and name in n:
                total += m.numel()
                params.append((n, m.numel()))
        else:
            if m.requires_grad:
                total += m.numel()
                params.append((n, m.numel()))
    return total, params

def _bytes_to_mb(x): 
    return float(x) / (1024**2)

def cali_flat_quant(args, model, dataloader, dev, logger):
    if torch.cuda.is_available():
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.empty_cache()

    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # check trainable parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # activate AMP
    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16 if isinstance(model, transformers.LlamaForCausalLM) else torch.bfloat16
        traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)

    # move embedding layer and first layer to target device
    layers = model.model.layers
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)

    # catch the first layer input
    if args.offload:
        inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu')
    else:
        inps = torch.zeros(
            (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
        )
    cache = {"i": 0}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            #inps[cache["i"]] = inp
            if args.offload:
                inps[cache["i"]].copy_(inp.squeeze(0).to('cpu', dtype=dtype))
            else:
                inps[cache["i"]].copy_(inp.squeeze(0))
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError
    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                sample = batch[0]
                model(sample.to(dev))
            except ValueError:
                pass
    position_ids = cache["position_ids"]
    attention_mask = cache["attention_mask"]
    if attention_mask is not None:
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).to(dev).float()
    else:
        attention_mask_batch = None
    position_ids = position_ids.to(dev) if position_ids is not None else None
    
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    # raise ValueError("Only support for llama-2/Llama-3/qwen-2 now")
    torch.cuda.empty_cache()

    # same input of first layer for fp model and quant model
    fp_inps = inps   # take output of fp model as input
    if args.offload:
        fp_outs = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size),
                            dtype=dtype, device='cpu')
    else:
        fp_outs = torch.zeros_like(inps, dtype=dtype, device=dev)  # take output of fp model as input
    fp_outs.zero_()  

    loss_func = torch.nn.MSELoss()
    # start training
    flat_parameters = {}
    num_train_layer = len(layers)
    mse_dict = {}
    for i in range(num_train_layer):
        if not i == 0:
            logger.info(f"========= Layer {i} =========")
        dtype_dict = {}
        layer = layers[i].to(dev)
        for name, param in layer.named_parameters():
            dtype_dict[name] = param.dtype
        with torch.no_grad():
            layer.float()

        layer.self_attn._ori_mode = True
        layer.mlp._ori_mode = True
        with torch.no_grad():
            if args.offload:
                for off in range(0, args.nsamples, args.cali_bsz):          
                    bs = min(args.cali_bsz, args.nsamples - off)              
                    x = fp_inps[off:off+bs].to(dev)  
                    if attention_mask_batch is None:                          
                        am = None                                      
                    else:                                               
                        am = attention_mask_batch if bs == args.cali_bsz else attention_mask.repeat(bs, 1, 1, 1).to(dev).float() 
                    y = layer(x, attention_mask=am, position_ids=position_ids)[0] 
                    fp_outs[off:off+bs].copy_(y.detach().to('cpu', dtype = dtype))
            else:
                for j in range(args.nsamples):
                    fp_outs[j] = layer(fp_inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layer.self_attn._ori_mode = False
        layer.mlp._ori_mode = False
        if not args.no_apply_trans:
            if args.diag_init == "sq_style":
                layer.self_attn.init_diag_scale(alpha=args.diag_alpha)
                layer.mlp.init_diag_scale(alpha=args.diag_alpha)
            elif args.diag_init == "one_style":
                pass
            else:
                raise NotImplementedError

        layer = layer.to(dev)
        set_require_grad_all(layer, False)
        trained_params, paras_name = [], []
        if args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.linear", ]), "lr": args.flat_lr})
            paras_name.append("trans.linear")
        if args.add_diag:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["trans.diag_scale", ]), "lr": args.flat_lr})
            paras_name.append("trans.diag_scale")
        if args.lwc:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_w", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_w")
        if args.lac:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["clip_factor_a", ]), "lr": args.flat_lr * 10})
            paras_name.append("clip_factor_a")

        if args.learn_weight:
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["learnable_weight", ]), "lr": args.weight_lr})
            paras_name.append("weight")

            trained_params.append({"params": get_n_set_parameters_byname(layer, ["input_layernorm.weight", ]), "lr": args.weight_lr })
            paras_name.append("input_layernorm")
            
            trained_params.append({"params": get_n_set_parameters_byname(layer, ["post_attention_layernorm.weight", ]), "lr": args.weight_lr })
            paras_name.append("post_attention_layernorm")

        if args.learn_scale:
            trained_params.append({"params": get_n_set_parameters_byname(layer, [".scale", ]), "lr": args.weight_lr * 10})
            paras_name.append("scale")

            if args.w_asym:
                trained_params.append({"params": get_n_set_parameters_byname(layer, [".zero", ]), "lr": args.weight_lr * 10})
                paras_name.append("zero")

        accumulate_steps = args.cali_bsz_accumulate_step
        optimizer = torch.optim.AdamW(trained_params)
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * (args.nsamples // (args.cali_bsz * accumulate_steps)), eta_min=args.flat_lr * 1e-3)
        if args.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=16)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
        else:
            scheduler = scheduler_main
        # check_params_grad(layer)
        # set_quantizer_state(layer, False)
        if i == 0:
            trainable_number, trainable_params = trainable_parameters_num(layer)
            logger.info(f"trainable parameter number: {trainable_number}")
            logger.info(f"trainable parameter name:")
            for name, number in trainable_params:
                logger.info(f"{name}: {number}")
            logger.info(f"========= Layer {i} =========")

        for epoch in range(args.epochs):
            if epoch == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(dev)
                    torch.cuda.reset_peak_memory_stats(dev)

            mse = 0
            start_tick = time.time()
            with traincast():
                iter = 0
                optimizer.zero_grad()

                for off in range(0, args.nsamples, args.cali_bsz):            
                    bs = min(args.cali_bsz, args.nsamples - off)             
                    x = fp_inps[off:off+bs].to(dev, non_blocking=True)      
                    y_ref = fp_outs[off:off+bs].to(dev, non_blocking=True) 
                    am = None if attention_mask_batch is None else (attention_mask_batch if bs == args.cali_bsz else attention_mask.repeat(bs,1,1,1).to(dev, non_blocking=True).float()) 
                    quant_out = layer(x, attention_mask=am, position_ids=position_ids)[0] 
                    if torch.isnan(quant_out).any():
                        logger.warning(f"NaN detected in layer {i}, epoch {epoch}")
                        for name, param in layer.named_parameters():
                            if param.requires_grad and torch.isnan(param).any():
                                logger.warning(f"NaN in parameter: {name}")
                    loss = loss_func(y_ref, quant_out)
                    mse += loss.detach().cpu()
                    if loss == 0:
                        print("loss = 0!")
                        import pdb; pdb.set_trace()
                    loss = loss / accumulate_steps
                    loss = loss / loss.clone().detach().clamp_min(1e-12)
                    loss.backward()
                    if (iter + 1) % accumulate_steps == 0 or off + bs >= args.nsamples:
                        optimizer.step()
                        if scheduler is not None: 
                            scheduler.step()
                        optimizer.zero_grad()
                    iter += 1

            if epoch == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize(dev)
                    peak_alloc = torch.cuda.max_memory_allocated(dev)
                    peak_resvd = torch.cuda.max_memory_reserved(dev)
                    logger.info(f"[MEM] layer {i} epoch {epoch} peak_alloc={_bytes_to_mb(peak_alloc):.1f}MB "
                                f"peak_resvd={_bytes_to_mb(peak_resvd):.1f}MB")
            cur_flat_lr = optimizer.state_dict()['param_groups'][0]['lr']
            if args.learn_scale:
                cur_weight_lr = optimizer.state_dict()['param_groups'][-1]['lr']
            
            if args.learn_scale:
                logger.info(f"layer {i} lwc lac iter {epoch}, flat_lr {cur_flat_lr:.8f}  weight_lr {cur_weight_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse / accumulate_steps:.8f}, mean_mse: {mse / iter :.8f}" )
            else:
                logger.info(f"layer {i} lwc lac iter {epoch}, flat_lr {cur_flat_lr:.8f}  time {time.time() - start_tick:.6f}s, mse: {mse / accumulate_steps:.8f}, mean_mse: {mse / iter :.8f}" )

        fp_inps, fp_outs = fp_outs, fp_inps
        layers[i] = layer.to(dtype=torch.float16, device="cpu")
        #layers[i] = layer.to("cpu")
        cur = get_paras_dict_by_name(layer, required_names=paras_name)
        cur = {k: v.detach().cpu().clone() for k, v in cur.items()}
        torch.save(cur, os.path.join(args.exp_dir, f"flat_parameters.pth"))
        del cur
        logger.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"flat_parameters.pth")))
        try: del optimizer
        except: pass
        try: del scheduler, scheduler_main, scheduler_warmup
        except: pass
        for name, param in layer.named_parameters():
            param.requires_grad = False
            if name in dtype_dict.keys():
                param.data = param.to(dtype_dict[name])
        del layer
        gc.collect()
        torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    torch.cuda.empty_cache()
    model.config.use_cache = use_cache
    return model

