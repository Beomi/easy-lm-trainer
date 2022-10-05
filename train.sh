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

# Ployglot-Ko 3.8B 사용 예시
python main.py \
    --model_name='EleutherAI/polyglot-ko-3.8b' \
    --train_file_path='./data/train.csv' \
    --test_file_path='./data/test.csv' \
    --num_train_epochs=5 \
    --data_text_column='text'
