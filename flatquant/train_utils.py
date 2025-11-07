import os
import time
import gc
import math
import functools
from contextlib import nullcontext
from typing import Optional
from tqdm import tqdm
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import transformers

from torch.nn.parallel import DistributedDataParallel as DDP
from flatquant.function_utils import set_require_grad_all, get_n_set_parameters_byname, get_paras_dict_by_name
from flatquant.utils import DistEnv
from flatquant.gptq_utils import rtn_fwrd, _fwrd
from flatquant.flat_utils import (
    reparameterize_block, save_quantized_weights_with_safetensors_block, save_misc_weights_with_safetensors_block
)

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

def _unwrap_module(module):
    if isinstance(module, DDP):
        return module.module
    return module

def print_cpu_memory_usage(message: str):
    import psutil
    memory_info = psutil.virtual_memory()
    used_memory = memory_info.used
    total_memory = memory_info.total
    print(f"{message}: {used_memory / 1024 ** 3:.1f} GB / {total_memory / 1024 ** 3:.1f} GB")

def cali_flat_quant(args, model, dataloader, dev, logger, dist_env: Optional[DistEnv] = None):
    dist_enabled = dist_env is not None and getattr(dist_env, "world_size", 1) > 1
    device = dist_env.device if dist_enabled else dev
    rank_zero = (not dist_enabled) or dist_env.rank == 0

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    model.eval()

    if args.blockwise_save:
        assert args.quantized_save, "blockwise saving requires quantized_save to be enabled"
        if dist_enabled:
            dist.barrier()
        if rank_zero:
            save_misc_weights_with_safetensors_block(args, model)
            logger.info("saved misc weights at {}".format(args.exp_dir))
        if dist_enabled:
            dist.barrier()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    for name, param in model.named_parameters():
        param.requires_grad = False

    if args.deactive_amp:
        dtype = torch.float32
        traincast = nullcontext
    else:
        dtype = torch.float16 if isinstance(model, transformers.LlamaForCausalLM) else torch.bfloat16
        traincast = functools.partial(torch.amp.autocast, device_type="cuda", dtype=dtype)

    layers = model.model.layers
    layers[0] = layers[0].to(device)
    model.model.embed_tokens = model.model.embed_tokens.to(device)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(device)

    total_nsamples = min(args.nsamples, len(dataloader))

    ddp_size = dist_env.ddp_size if dist_enabled else 1
    dp_rank = dist_env.dp_rank if dist_enabled else 0
    dp_group = dist_env.dp_group if dist_enabled else None

    # we tuned learning rate for batch size 4, so we need to scale the learning rate for other batch sizes
    bsz_scale = args.cali_bsz * args.cali_bsz_accumulate_step / 4.0
    args.flat_lr = args.flat_lr * bsz_scale
    args.weight_lr = args.weight_lr * bsz_scale
    args.scale_lr = args.scale_lr * bsz_scale
    msg = f"scaled learning rate for batch size {args.cali_bsz * args.cali_bsz_accumulate_step}: flat_lr {args.flat_lr}"
    if args.learn_weight:
        msg += f", weight_lr {args.weight_lr}"
    if args.learn_scale:
        msg += f", scale_lr {args.scale_lr}"
    logger.info(msg)

    samples_per_rank = total_nsamples if ddp_size == 1 else math.ceil(total_nsamples / ddp_size)
    local_indices = [(idx * ddp_size + dp_rank) % total_nsamples for idx in range(samples_per_rank)]
    local_nsamples = len(local_indices)

    base_accumulate = args.cali_bsz_accumulate_step
    accumulate_steps = base_accumulate
    if dist_enabled and ddp_size > 1:
        accumulate_steps = max(1, math.ceil(base_accumulate / ddp_size))
        if rank_zero and accumulate_steps * ddp_size != base_accumulate:
            raise ValueError(
                "cali_bsz_accumulate_step (%d) is not divisible by ddp_size (%d); "
                "using per-rank accumulation=%d which changes the effective gradient accumulation.",
                base_accumulate,
                ddp_size,
                accumulate_steps,
            )

    if dist_enabled and ddp_size > 1 and rank_zero and total_nsamples % ddp_size != 0:
        logger.warning("Calibration samples are re-used on some ranks to keep step counts aligned; consider increasing nsamples or adjusting ddp_size.")

    storage_device = 'cpu' if args.offload else device
    inps = torch.zeros((local_nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=storage_device)
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            if cache["i"] >= local_nsamples:
                raise ValueError
            tensor = inp.squeeze(0)
            if args.offload:
                inps[cache["i"]].copy_(tensor.to('cpu', dtype=dtype))
            else:
                inps[cache["i"]].copy_(tensor)
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for idx in local_indices:
            if cache["i"] >= local_nsamples:
                break
            sample = dataloader[idx][0]
            try:
                model(sample.to(device))
            except ValueError:
                pass

    attention_mask = cache.get("attention_mask")
    position_ids = cache.get("position_ids")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None
    position_ids = position_ids.to(device) if position_ids is not None else None

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    fp_inps = inps
    if args.offload:
        fp_outs = torch.zeros((local_nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device='cpu')
    else:
        fp_outs = torch.zeros_like(inps, dtype=dtype, device=device)
    fp_outs.zero_()

    loss_func = torch.nn.MSELoss()
    # start training
    flat_parameters = {}
    num_train_layer = len(layers)
    mse_dict = {}
    for i in range(num_train_layer):
        if not i == 0 and rank_zero:
            logger.info(f"========= Layer {i} =========")
        layer = layers[i]
        dtype_dict = {name: param.dtype for name, param in layer.named_parameters()}
        layer = layer.to(device=device, dtype=torch.float32)

        grad_enable_tags = []
        if args.cali_trans:
            grad_enable_tags.append("trans.linear")
        if args.add_diag:
            grad_enable_tags.append("trans.diag_scale")
        if args.lwc:
            grad_enable_tags.append("clip_factor_w")
        if args.lac:
            grad_enable_tags.append("clip_factor_a")
        if args.learn_weight:
            grad_enable_tags.extend([
                "linear.weight",
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
            ])
        if args.learn_scale:
            grad_enable_tags.append(".scale")
            if args.w_asym:
                grad_enable_tags.append(".zero")

        for tag in grad_enable_tags:
            for name, param in layer.named_parameters():
                if tag in name:
                    param.requires_grad = True

        has_trainable = any(param.requires_grad for param in layer.parameters())

        wrapped_layer = layer
        if dist_enabled and has_trainable and ddp_size > 1:
            wrapped_layer = DDP(
                wrapped_layer, device_ids=[device.index], process_group=dp_group, broadcast_buffers=False
            )
        layer = wrapped_layer
        module = _unwrap_module(layer)

        module.self_attn._ori_mode = True
        module.mlp._ori_mode = True
        with torch.no_grad():
            for off in tqdm(range(0, local_nsamples, args.cali_bsz), desc=f"Calculating fp_outs for layer {i}"):
                bs = min(args.cali_bsz, local_nsamples - off)
                x = fp_inps[off:off+bs]
                if x.device != device:
                    x = x.to(device, non_blocking=device.type == 'cuda')
                if attention_mask_batch is None:
                    am = None
                else:
                    if bs == args.cali_bsz:
                        am = attention_mask_batch
                    else:
                        am = attention_mask.repeat(bs, 1, 1, 1).to(device, non_blocking=device.type == 'cuda').float()
                y = layer(x, attention_mask=am, position_ids=position_ids)[0]
                if args.offload:
                    fp_outs[off:off+bs].copy_(y.detach().to('cpu', dtype=dtype))
                else:
                    fp_outs[off:off+bs] = y.detach()
        module.self_attn._ori_mode = False
        module.mlp._ori_mode = False
        if not args.no_apply_trans:
            if args.diag_init == "sq_style":
                module.self_attn.init_diag_scale(alpha=args.diag_alpha)
                module.mlp.init_diag_scale(alpha=args.diag_alpha)
            elif args.diag_init == "one_style":
                pass
            else:
                raise NotImplementedError

        set_require_grad_all(module, False)
        trained_params, paras_name = [], []
        flat_param, clip_param, weight_param, scale_param = [], [], [], []
        if args.cali_trans:
            trained_params.append({"params": get_n_set_parameters_byname(module, ["trans.linear", ]), "lr": args.flat_lr, "tag": "trans.linear"})
            paras_name.append("trans.linear")
            flat_param.append("trans.linear")
        if args.add_diag:
            trained_params.append({"params": get_n_set_parameters_byname(module, ["trans.diag_scale", ]), "lr": args.flat_lr, "tag": "trans.diag_scale"})
            paras_name.append("trans.diag_scale")
            flat_param.append("trans.diag_scale")
        if args.lwc:
            trained_params.append({"params": get_n_set_parameters_byname(module, ["clip_factor_w", ]), "lr": args.flat_lr * 10, "tag": "clip_factor_w"})
            paras_name.append("clip_factor_w")
            clip_param.append("clip_factor_w")
        if args.lac:
            trained_params.append({"params": get_n_set_parameters_byname(module, ["clip_factor_a", ]), "lr": args.flat_lr * 10, "tag": "clip_factor_a"})
            paras_name.append("clip_factor_a")
            clip_param.append("clip_factor_a")

        if args.learn_weight:
            trained_params.append({"params": get_n_set_parameters_byname(module, ["linear.weight", ]), "lr": args.weight_lr, "tag": "linear"})
            paras_name.append("linear")
            weight_param.append("linear")

            trained_params.append({"params": get_n_set_parameters_byname(module, ["input_layernorm.weight", ]), "lr": args.weight_lr, "tag": "input_layernorm"})
            paras_name.append("input_layernorm")
            weight_param.append("input_layernorm")
            
            trained_params.append({"params": get_n_set_parameters_byname(module, ["post_attention_layernorm.weight", ]), "lr": args.weight_lr, "tag": "post_attention_layernorm"})
            paras_name.append("post_attention_layernorm")
            weight_param.append("post_attention_layernorm")

        if args.learn_scale:
            trained_params.append({"params": get_n_set_parameters_byname(module, [".scale", ]), "lr": args.scale_lr, "tag": "scale"})
            paras_name.append("scale")
            scale_param.append("scale")

            if args.w_asym:
                trained_params.append({"params": get_n_set_parameters_byname(module, [".zero", ]), "lr": args.scale_lr, "tag": "zero"})
                paras_name.append("zero")
                scale_param.append("zero")

        schedule_steps = max(1, math.ceil(local_nsamples / (args.cali_bsz * accumulate_steps)))
        optimizer = torch.optim.AdamW(trained_params)
        empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.flat_lr)
        empty_optimizer_0 = torch.optim.AdamW([torch.tensor(0)], lr=args.flat_lr * 10)
        empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
        empty_optimizer_3 = torch.optim.AdamW([torch.tensor(0)], lr=args.scale_lr)
        group_idx = { g.get("tag", f"group{i}"): i for i, g in enumerate(optimizer.param_groups) }
        scheduler_main = torch.optim.lr_scheduler.CosineAnnealingLR(empty_optimizer_1, T_max=args.epochs * schedule_steps, eta_min=args.flat_lr * 1e-3)
        scheduler_clip = torch.optim.lr_scheduler.CosineAnnealingLR(empty_optimizer_0, T_max=args.epochs * schedule_steps, eta_min=args.flat_lr * 10 * 1e-3)
        scheduler_weight = torch.optim.lr_scheduler.CosineAnnealingLR(empty_optimizer_2, T_max=args.epochs * schedule_steps, eta_min=args.weight_lr / 20)
        scheduler_scale = torch.optim.lr_scheduler.CosineAnnealingLR(empty_optimizer_3, T_max=args.epochs * schedule_steps, eta_min=args.scale_lr / 20)
        if args.warmup:
            scheduler_warmup = torch.optim.lr_scheduler.LinearLR(empty_optimizer_1, start_factor=0.01, total_iters=16)
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup, scheduler_main])
            scheduler_warmup_2 = torch.optim.lr_scheduler.LinearLR(empty_optimizer_0, start_factor=0.01, total_iters=16)
            scheduler_clip = torch.optim.lr_scheduler.ChainedScheduler([scheduler_warmup_2, scheduler_clip])
        else:
            scheduler = scheduler_main
        # check_params_grad(layer)
        # set_quantizer_state(layer, False)
        if i == 0 and rank_zero:
            trainable_number, trainable_params = trainable_parameters_num(module)
            logger.info(f"trainable parameter number: {trainable_number}")
            logger.info(f"trainable parameter name:")
            for name, number in trainable_params:
                logger.info(f"{name}: {number}")
            logger.info(f"========= Layer {i} =========")

        for epoch in range(args.epochs):
            if epoch == 0 and torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)

            mse = 0.0
            start_tick = time.time()
            with traincast():
                iter = 0
                optimizer.zero_grad()

                for off in range(0, local_nsamples, args.cali_bsz):
                    bs = min(args.cali_bsz, local_nsamples - off)
                    x = fp_inps[off:off+bs]
                    y_ref = fp_outs[off:off+bs]
                    if x.device != device:
                        x = x.to(device, non_blocking=device.type == 'cuda')
                    if y_ref.device != device:
                        y_ref = y_ref.to(device, non_blocking=device.type == 'cuda')
                    if attention_mask_batch is None:
                        am = None
                    else:
                        if bs == args.cali_bsz:
                            am = attention_mask_batch
                        else:
                            am = attention_mask.repeat(bs,1,1,1).to(device, non_blocking=device.type == 'cuda').float()
                    
                    sync_now = ((iter + 1) % accumulate_steps == 0 or off + bs >= local_nsamples)
                    if isinstance(layer, DDP) and not sync_now:
                        ctx = layer.no_sync()
                    else:
                        ctx = nullcontext()

                    with ctx:
                        quant_out = layer(x, attention_mask=am, position_ids=position_ids)[0]
                        if torch.isnan(quant_out).any():
                            if rank_zero:
                                logger.warning(f"NaN detected in layer {i}, epoch {epoch}")
                            for name, param in module.named_parameters():
                                if param.requires_grad and torch.isnan(param).any():
                                    if rank_zero:
                                        logger.warning(f"NaN in parameter: {name}")
                        loss = loss_func(y_ref, quant_out)
                        mse += loss.detach().float().item()
                        if loss == 0:
                            print("loss = 0!")
                            import pdb; pdb.set_trace()
                        loss = loss / accumulate_steps
                        loss = loss / loss.clone().detach().clamp_min(1e-12)
                        loss.backward()

                    if sync_now:
                        optimizer.step()
                        if scheduler is not None: 
                            scheduler.step()
                            scheduler_clip.step()
                            scheduler_weight.step()
                            scheduler_scale.step()
                            for tag in flat_param:
                                optimizer.param_groups[group_idx[tag]]['lr'] = scheduler.get_last_lr()[0]
                            for tag in clip_param:
                                optimizer.param_groups[group_idx[tag]]['lr'] = scheduler_clip.get_last_lr()[0]
                            for tag in weight_param:
                                optimizer.param_groups[group_idx[tag]]['lr'] = scheduler_weight.get_last_lr()[0]
                            for tag in scale_param:
                                optimizer.param_groups[group_idx[tag]]['lr'] = scheduler_scale.get_last_lr()[0]
                        optimizer.zero_grad()
                    iter += 1

            if epoch == 0 and torch.cuda.is_available() and device.type == 'cuda':
                torch.cuda.synchronize(device)
                peak_alloc = torch.cuda.max_memory_allocated(device)
                peak_resvd = torch.cuda.max_memory_reserved(device)
                if rank_zero:
                    logger.info(f"[MEM] layer {i} epoch {epoch} peak_alloc={_bytes_to_mb(peak_alloc):.1f}MB "
                                f"peak_resvd={_bytes_to_mb(peak_resvd):.1f}MB")
            cur_flat_lr = optimizer.state_dict()['param_groups'][0]['lr']
            if args.learn_weight:
                cur_weight_lr = optimizer.state_dict()['param_groups'][group_idx["linear"]]['lr']
            if args.learn_scale:
                cur_scale_lr = optimizer.state_dict()['param_groups'][group_idx["scale"]]['lr']
            
            mse_tensor = torch.tensor(mse, device=device if device.type == 'cuda' else 'cpu')
            if dist_enabled and ddp_size > 1:
                dist.all_reduce(mse_tensor, op=dist.ReduceOp.SUM, group=dp_group)
                mse_value = mse_tensor.item() / ddp_size
            else:
                mse_value = mse_tensor.item()

            if rank_zero:
                if args.learn_weight:
                    if args.learn_scale:
                        logger.info(f"layer {i} lwc lac iter {epoch}, flat_lr {cur_flat_lr:.8f}, weight_lr {cur_weight_lr:.8f}, scale_lr {cur_scale_lr:.8f}, time {time.time() - start_tick:.6f}s, mse: {mse_value / accumulate_steps:.8f}, mean_mse: {mse_value / iter :.8f}")
                    else:
                        logger.info(f"layer {i} lwc lac iter {epoch}, flat_lr {cur_flat_lr:.8f}, weight_lr {cur_weight_lr:.8f}, time {time.time() - start_tick:.6f}s, mse: {mse_value / accumulate_steps:.8f}, mean_mse: {mse_value / iter :.8f}")
                else:
                    if args.learn_scale:
                        logger.info(f"layer {i} lwc lac iter {epoch}, flat_lr {cur_flat_lr:.8f}, scale_lr {cur_scale_lr:.8f}, time {time.time() - start_tick:.6f}s, mse: {mse_value / accumulate_steps:.8f}, mean_mse: {mse_value / iter :.8f}")
                    else:
                        logger.info(f"layer {i} lwc lac iter {epoch}, flat_lr {cur_flat_lr:.8f}, time {time.time() - start_tick:.6f}s, mse: {mse_value / accumulate_steps:.8f}, mean_mse: {mse_value / iter :.8f}")

        fp_inps, fp_outs = fp_outs, fp_inps

        optimizer.zero_grad(set_to_none=True)
        x = y_ref = quant_out = am = None
        del optimizer, trained_params, scheduler, scheduler_main
        del empty_optimizer_1, empty_optimizer_0, empty_optimizer_2, empty_optimizer_3

        if rank_zero and args.save_matrix:
            cur = get_paras_dict_by_name(module, required_names=paras_name)
            cur = {k: v.detach().cpu().clone() for k, v in cur.items()}
            if not dist_enabled or rank_zero:
                torch.save(cur, os.path.join(args.exp_dir, f"flat_parameters.pth"))
                logger.info("saved paramaters at {}".format(os.path.join(args.exp_dir, f"flat_parameters.pth")))
            del cur
        
        if rank_zero:
            print_cpu_memory_usage(f"before saving the block")
            for name, param in module.named_parameters():
                param.requires_grad = False
                if name in dtype_dict.keys():
                    param.data = param.to(dtype_dict[name])
            if args.blockwise_save:
                reparameterize_block(module)
                if args.w_bits < 16:
                    if not args.learn_scale:
                        if args.gptq: # GPTQ Weight Quantization
                            raise NotImplementedError("blockwise saving with GPTQ is not supported yet")
                        else: # RTN Weight Quantization
                            quantizers = rtn_fwrd(module, device, args, layer_id=i)
                    else:
                        quantizers = _fwrd(module, device, args, layer_id=i)
                save_quantized_weights_with_safetensors_block(args, module, quantizers, layer_id=i)
                layers[i] = module.to_empty(device="meta")
            else:
                layers[i] = module.to(device="cpu")
            print_cpu_memory_usage(f"after saving the block")
        else:
            layers[i] = module.to_empty(device="meta")

        del module
        gc.collect()
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.empty_cache()

    del inps, fp_inps, fp_outs
    gc.collect()
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.empty_cache()
    if dist_enabled:
        dist.barrier()
    if rank_zero:
        total_size = 0
        weight_map = {}
        for file in os.listdir(args.exp_dir):
            if file.endswith(".index.json"):
                with open(os.path.join(args.exp_dir, file), "r") as f:
                    index = json.load(f)
                    total_size += index["metadata"]["total_size"]
                    weight_map.update(index["weight_map"])
        with open(os.path.join(args.exp_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, f, indent=2)
        
        for file in os.listdir(args.exp_dir):
            if file.endswith(".index.json") and file != "model.safetensors.index.json":
                os.remove(os.path.join(args.exp_dir, file))
    if dist_enabled:
        dist.barrier()
    model.config.use_cache = use_cache
    return model
