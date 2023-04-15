# Easy LM Trainer

Huggingface Transformers를 사용해 LM을 학습할 때, 단순 LM(CLM) 학습 작업을 보다 쉽게 시작하기 위한 Boilerplate 프로젝트.

## 환경

- CPython 3.10+
- PyTorch는 CUDA 환경에 맞게 설치하기 (1.12.1 이상)

```bash
pip install -r requirements.txt
pip install -U deepspeed # Deepspeed branch 한정
```

## 실행

```bash
./train.sh
```

- 예시 환경은 RAM 1TB, GPU A100 40GB x4장 환경에서 실험 
- CUDA 11.6/11.7
- PyTorch 2.0
- KoAlpaca Dataset을 학습 데이터로 사용함
- DeepSpeed ZeRO3, Optimizer와 Parameter 모두 CPU Offload
- Seq len 1024
- Max batch size 1 (per GPU)
- GPU당 약 27GB vram 사용 (= V100 32G에서도 사용 가능할 것으로 예상)
