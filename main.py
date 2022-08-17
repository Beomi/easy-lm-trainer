import random
from datetime import date
from pathlib import Path
from pprint import pprint

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
)


class SimpleArgumentParser(Tap):
    use_projectwise_cache: bool = False
    use_random_uuid: bool = True
    block_size: int = 256
    batch_size: int = 8
    model_name: str
    model_revision: str = ""
    bos_token: str = ""
    eos_token: str = ""
    unk_token: str = ""
    pad_token: str = ""
    mask_token: str = ""
    data_file_path: Path
    data_text_column: str = "text"
    group_text: bool = False
    force_retrain: bool = False
    num_train_epochs: int = 1
    learning_rate: float = 0.0001
    bf16: bool = False
    fp16: bool = False
    optimizer: str = "adafactor"
    gradient_accumulation_steps: int = 1
    do_test_generate: bool = True
    random_seed: int = 42


def main(args):
    if args.use_random_uuid:
        uid = OrderedUUID()
        uid = "-".join(str(uid).split("-")[:2])
        print(f"{uid=}")
    else:
        uid = ""

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.model_revision if args.model_revision else None,
        bos_token=None if not args.bos_token else args.bos_token,
        eos_token=None if not args.eos_token else args.eos_token,
        unk_token=None if not args.unk_token else args.unk_token,
        pad_token=None if not args.pad_token else args.pad_token,
        mask_token=None if not args.mask_token else args.mask_token,
        model_max_length=args.block_size,
    )

    def tokenize_function(examples):
        if args.group_text:
            s = tokenizer(examples[args.data_text_column])
        else:
            s = tokenizer(
                examples[args.data_text_column],
                padding="max_length",
                max_length=args.block_size,
                truncation=True,
            )
        if not args.group_text:
            s["labels"] = s["input_ids"].copy()
        return s

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // args.block_size) * args.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + args.block_size]
                for i in range(0, total_length, args.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    today = date.today().isoformat()

    if str(args.data_file_path).endswith(".csv"):
        df = pd.read_csv(args.data_file_path)
    elif str(args.data_file_path).endswith(".json"):
        df = pd.read_json(args.data_file_path)
    elif str(args.data_file_path).endswith(".xlsx"):
        df = pd.read_excel(args.data_file_path)
    elif str(args.data_file_path).endswith(".tsv"):
        df = pd.read_csv(args.data_file_path, sep="\t")
    elif str(args.data_file_path).endswith(".txt"):
        df = pd.read_csv(args.data_file_path, sep="\t")
    else:
        raise ValueError(f"Unknown file format: {args.data_file_path}")

    assert args.data_text_column in df.columns

    print(df.head(1))

    SAVE_PATH = f"./models/{today}_{uid}/"
    if not args.force_retrain:  # Retrain 시에는 무시하고 Overwrite
        if (Path(SAVE_PATH) / "pytorch_model.bin").exists():
            print("이미 모델 있음. 지우고 진행하기.")
            return  # 이미 학습됨.

    train, test = train_test_split(df, random_state=42, test_size=0.1)
    datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(train[[args.data_text_column]]),
            "test": Dataset.from_pandas(test[[args.data_text_column]]),
        }
    )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=[args.data_text_column],
    )

    if args.group_text:
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            batch_size=4,
            num_proc=2,
        )
    else:
        lm_datasets = tokenized_datasets

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.model_revision if args.model_revision else None,
        pad_token_id=tokenizer.eos_token_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True if torch.cuda.is_available() else False,
    )

    training_args = TrainingArguments(
        SAVE_PATH,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_first_step=True,
        logging_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        learning_rate=args.learning_rate,
        # weight_decay=0.001,
        optim=args.optimizer,
        push_to_hub=False,
        per_device_train_batch_size=args.batch_size,
        bf16=args.bf16,
        fp16=args.fp16,
        num_train_epochs=args.num_train_epochs,
        # deepspeed=ds_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
    )
    try:
        trainer.train()
    except KeyboardInterrupt:
        return model, tokenizer
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
