# Kakao KoGPT 6B 사용 예시
# python main.py \
#     --model_name='kakaobrain/kogpt' \
#     --model_revision='KoGPT6B-ryan1.5b-float16' \
#     --bos_token='[BOS]' \
#     --eos_token='[EOS]' \
#     --unk_token='[UNK]' \
#     --pad_token='[PAD]' \
#     --mask_token='[MASK]' \
#     --train_file_path='./data/train.csv' \
#     --test_file_path='./data/test.csv' \
#     --num_train_epochs=10 \
#     --data_text_column='text'

# # Ployglot-Ko 3.8B 사용 예시
# torchrun --nproc_per_node=4 --master_port=34321 main.py \
#     --model_name='EleutherAI/polyglot-ko-5.8b' \
#     --train_file_path='data/text_ko_alpaca_data.jsonl' \
#     --num_train_epochs=1 \
#     --data_text_column='text' \
#     --block_size=256 \
#     --batch_size=1 \
#     --fp16=True \
#     --fsdp='auto_wrap' \
#     --fsdp_config=fsdp_config.json \
#     --deepspeed=ds_config.json
#     # --fsdp_transformer_layer_cls_to_wrap='GPTNeoXLayer' \

# Ployglot-Ko 12.8B 4GPU 사용 예시
torchrun --nproc_per_node=4 --master_port=34321 main.py \
    --model_name='EleutherAI/polyglot-ko-12.8b' \
    --train_file_path='data/text_ko_alpaca_data.jsonl' \
    --num_train_epochs=1 \
    --data_text_column='text' \
    --block_size=1024 \
    --batch_size=1 \
    --bf16=True \
    --group_text=True \
    --deepspeed=ds_zero3.json
