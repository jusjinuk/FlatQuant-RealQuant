import os
import pickle
import datasets
import random
import transformers

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        traindata = traindata.filter(lambda x: len(x) > 0)
        traindata = traindata.map(lambda x : {'text': x['text'].strip()})
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seed, seqlen, tokenizer, eval_mode=False):
    if eval_mode:
        testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        # random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_pile(nsamples, seed, seqlen, tokenizer):
    traindata = datasets.load_dataset("pile-val-backup", split="validation")
    trainenc = tokenizer("\n\n".join(traindata['text'][:1000]), return_tensors='pt')
    # random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader

def get_redpajama(nsamples, seed, seqlen, tokenizer):
    traindata = datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split='train', trust_remote_code=True)['text']

    #random.seed(seed)
    selected_indices = set()
    trainloader = []
    while len(trainloader) < nsamples:
        idx = random.randint(0, len(traindata) - 1)
        if idx in selected_indices:
            continue

        text = traindata[idx]
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]  # (T,)
        if ids.shape[0] < seqlen:
            continue

        start = random.randint(0, ids.shape[0] - seqlen)
        inp = ids[start:start + seqlen].unsqueeze(0)               # (1, S)
        tar = inp.clone()
        tar[:, :-1] = -100
        selected_indices.add(idx)

        trainloader.append((inp, tar))  # ← 위키/ptb/c4와 동일 포맷
    return trainloader


def get_loaders(
    args, name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False
):
    #cache_dir = os.path.join(args.cache_dir, name)
    #os.makedirs(cache_dir, exist_ok=True)
    #cached_dataset = os.path.join(cache_dir, "testset.pkl" if eval_mode else f"trainset-{nsamples}-{seed}.pkl")
    # if os.path.exists(cached_dataset):
    if False:
        print(f"Loading cached tokenized dataset at {cached_dataset}...")
        with open(cached_dataset, "rb") as f:
            dataset = pickle.load(f)
    else:
        if hf_token is None:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
        if 'wikitext2' in name:
            dataset = get_wikitext2(nsamples, seed, seqlen, tokenizer, eval_mode)
        elif 'ptb' in name:
            dataset = get_ptb_new(nsamples, seed, seqlen, tokenizer, eval_mode)
        elif 'c4' in name:
            dataset = get_c4_new(nsamples, seed, seqlen, tokenizer, eval_mode)
        elif 'pile' in name:
            dataset = get_pile(nsamples, seed, seqlen, tokenizer)
        elif 'redpajama' in name:
            dataset = get_redpajama(nsamples, seed, seqlen, tokenizer)
        # with open(cached_dataset, "wb") as f:
        #     print(f"Saving cached tokenized dataset at {cached_dataset}...")
        #     if 'c4' in name and eval_mode:
        #         dataset = dataset.input_ids
        #     pickle.dump(dataset, f)
    if 'c4' in name and eval_mode:
        dataset = dataset.input_ids
        dataset = TokenizerWrapper(dataset)
    return dataset
