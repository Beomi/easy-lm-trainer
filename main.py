import random
from datetime import date
from pathlib import Path
from pprint import pprint
from itertools import chain
from typing import Optional

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from ordereduuid import OrderedUUID
from sklearn.model_selection import train_test_split
from tap import Tap
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
    default_data_collator,
)


class SimpleArgumentParser(Tap):
    use_projectwise_cache: bool = False
    use_random_uuid: bool = True
    uuid: str = ""
    block_size: int = 1024
    batch_size: int = 8
    model_name: str
    model_revision: str = ""
    bos_token: str = ""
    eos_token: str = ""
    unk_token: str = ""
    pad_token: str = ""
    mask_token: str = ""
    train_file_path: str
    test_file_path: str = "" # Optional
    data_text_column: str = "text"
    group_text: bool = False
    force_retrain: bool = False
    num_train_epochs: int = 1
    learning_rate: float = 0.0001
    bf16: bool = False
    fp16: bool = False
    optimizer: str = "adafactor"
    gradient_accumulation_steps: int = 1
    do_test_generate: bool = False
    random_seed: int = 42
    do_eval: bool = False
    save_as_fp16: bool = True
    fsdp: list = []
    fsdp_config: str = ''
    deepspeed: str = ''


def main(args):
    if args.use_random_uuid:
        if args.uuid:
            raise Exception("`use_random_uuid` and `uuid` should not used together!")
        uid = OrderedUUID()
        uid = "-".join(str(uid).split("-")[:2])
        print(f"{uid=}")
    elif args.uuid:
        uid = args.uuid
    else:
        uid = ""
    

    arg_dict = {}

    if args.bos_token:
        arg_dict["bos_token"] = args.bos_token
    if args.eos_token:
        arg_dict["eos_token"] = args.eos_token
    if args.unk_token:
        arg_dict["unk_token"] = args.unk_token
    if args.pad_token:
        arg_dict["pad_token"] = args.pad_token
    if args.mask_token:
        arg_dict["mask_token"] = args.mask_token
    if args.block_size:
        arg_dict["model_max_length"] = args.block_size
    if args.model_revision:
        arg_dict["model_revision"] = args.model_revision

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision if args.model_revision else None,
        **arg_dict,
    )

    def tokenize_function(examples):
        if args.group_text:
            s = tokenizer(examples[args.data_text_column])
            return s
        else:
            s = tokenizer(
                examples[args.data_text_column],
                padding="max_length",
                max_length=args.block_size,
                truncation=True,
            )
            s["labels"] = s["input_ids"].copy()
            return s

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        print(f"{total_length=}")
        
        block_size = args.block_size

        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    today = date.today().isoformat()
    SAVE_PATH = f"./models/{today}_{uid}/"
    
    if args.test_file_path: # Test 파일 있는 경우
        if args.train_file_path.endswith(".csv"):
            df = pd.read_csv(args.train_file_path)
        elif args.train_file_path.endswith(".json"):
            df = pd.read_json(args.train_file_path)
        elif args.train_file_path.endswith(".xlsx"):
            df = pd.read_excel(args.train_file_path)
        elif args.train_file_path.endswith(".tsv"):
            df = pd.read_csv(args.train_file_path, sep="\t")
        elif args.train_file_path.endswith(".txt"):
            df = pd.read_csv(args.train_file_path, sep="\t")
        else:
            raise ValueError(f"Unknown file format: {args.train_file_path}")
            
        if args.test_file_path.endswith(".csv"):
            test_df = pd.read_csv(args.test_file_path)
        elif args.test_file_path.endswith(".json"):
            test_df = pd.read_json(args.test_file_path)
        elif args.test_file_path.endswith(".xlsx"):
            test_df = pd.read_excel(args.test_file_path)
        elif args.test_file_path.endswith(".tsv"):
            test_df = pd.read_csv(args.test_file_path, sep="\t")
        elif args.test_file_path.endswith(".txt"):
            test_df = pd.read_csv(args.test_file_path, sep="\t")
        else:
            raise ValueError(f"Unknown file format: {args.test_file_path}")

        assert args.data_text_column in df.columns
        assert args.data_text_column in test_df.columns

        print(df.head(1))
        print(test_df.head(1))
        
        if not args.force_retrain:  # Retrain 시에는 무시하고 Overwrite
            if (Path(SAVE_PATH) / "pytorch_model.bin").exists():
                print("이미 모델 있음. 지우고 진행하기.")
                return  # 이미 학습됨.

        train, test = df, test_df
        
    else: # Train 만 있는 경우
        if args.train_file_path.endswith(".csv"):
            df = pd.read_csv(args.train_file_path)
        elif args.train_file_path.endswith(".json"):
            df = pd.read_json(args.train_file_path)
        elif args.train_file_path.endswith(".jsonl"):
            df = pd.read_json(args.train_file_path, lines=True)
        elif args.train_file_path.endswith(".xlsx"):
            df = pd.read_excel(args.train_file_path)
        elif args.train_file_path.endswith(".tsv"):
            df = pd.read_csv(args.train_file_path, sep="\t")
        elif args.train_file_path.endswith(".txt"):
            df = pd.read_csv(args.train_file_path, sep="\t")
        else:
            raise ValueError(f"Unknown file format: {args.train_file_path}")

        assert args.data_text_column in df.columns

        print(df.head(1))
        
        if not args.force_retrain:  # Retrain 시에는 무시하고 Overwrite
            if (Path(SAVE_PATH) / "pytorch_model.bin").exists():
                print("이미 모델 있음. 지우고 진행하기.")
                return  # 이미 학습됨.

        if args.do_eval:
            train, test = train_test_split(df, random_state=42, test_size=0.01)
            train.to_json(args.train_file_path+f'.{uid}.train.json')
            test.to_json(args.train_file_path+f'.{uid}.test.json') # machine readable
            datasets = DatasetDict(
                {
                    "train": Dataset.from_pandas(train[[args.data_text_column]]),
                    "test": Dataset.from_pandas(test[[args.data_text_column]]),
                }
            )
        else:
            train = df
            datasets = DatasetDict(
                {
                    "train": Dataset.from_pandas(train[[args.data_text_column]]),
                }
            )
        
    datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(train[[args.data_text_column]]),
            # "test": Dataset.from_pandas(test[[args.data_text_column]]),
        }
    )
    print(datasets)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=[args.data_text_column],
    )
    print(tokenized_datasets)

    if args.group_text:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=4,
        )
    else:
        lm_datasets = tokenized_datasets
        
    def _get_model_load_type():
        if torch.cuda.is_available() and args.fp16:
            return torch.float16
        elif torch.cuda.is_available() and args.bf16:
            return torch.bfloa16
        else:
            return 'auto'

    training_args = TrainingArguments(
        SAVE_PATH,
        save_strategy="epoch",
        evaluation_strategy="epoch" if args.do_eval else "no",
        logging_first_step=True,
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True if args.do_eval else False,
        learning_rate=args.learning_rate,
        # weight_decay=0.001,
        # optim=args.optimizer,
        push_to_hub=False,
        per_device_train_batch_size=args.batch_size,
        # train_micro_batch_size_per_gpu=args.batch_size,
        bf16=args.bf16,
        fp16=args.fp16,
        num_train_epochs=args.num_train_epochs,
        deepspeed=args.deepspeed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # fsdp=args.fsdp,
        # fsdp_config=args.fsdp_config,
        # fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls_to_wrap,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.model_revision if args.model_revision else None,
        torch_dtype='auto',  #_get_model_load_type(),
        # low_cpu_mem_usage=True # Not available when Zero-3
    )

    trainer_args_dict = {}
    if args.group_text:
        trainer_args_dict['data_collator'] = default_data_collator

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets.get('test') if args.do_eval else None,
        **trainer_args_dict,
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        return model, tokenizer
    
    if args.save_as_fp16:
        model.half().save_pretrained(SAVE_PATH)
    else:
        model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("model saved to ", SAVE_PATH)

    if args.do_test_generate:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            framework="pt",
        )
        res = pipe(test.head()[args.data_text_column].to_list())
        pprint(res)
    return model, tokenizer


if __name__ == "__main__":
    args = SimpleArgumentParser(explicit_bool=True).parse_args()

    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    if args.use_projectwise_cache:
        import os

        os.environ["TRANSFORMERS_CACHE"] = "./.cache"

    from transformers import pipeline

    model, tokenizer = main(args)
