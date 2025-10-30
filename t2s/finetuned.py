import os
import torch
import json
import re
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    EarlyStoppingCallback,  # 과적합 방지를 위해 추가
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# bitsandbytes 관련 환영 메시지 끄기
os.environ["BITSANDBYTES_NOWELCOME"] = "1"


# --- 1단계: 모델 및 데이터셋 로드 (rope_scaling 에러 해결) ---
def setup_environment():
    """모델, 토크나이저, 데이터셋을 로드하고 설정합니다."""
    print("환경 설정 시작...")

    try:
        from huggingface_hub import whoami
        print(f"Hugging Face에 '{whoami()['name']}' 계정으로 로그인되어 있습니다.")
    except Exception:
        print("경고: Hugging Face 로그인이 확인되지 않았습니다. 터미널에서 'huggingface-cli login'을 실행해주세요.")

    spider_dataset = load_dataset("spider")
    print("Spider 데이터셋 로드 완료.")

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"모델 ID: {model_id} 로 설정을 시작합니다.")

    # 'rope_scaling' 에러 해결을 위해 모델 설정을 먼저 로드
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # RoPE scaling 설정을 라이브러리가 호환 가능한 간단한 형식으로 변경
    if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
        # 기존 factor 값을 유지하되, 간단한 형식으로 재구성
        original_factor = config.rope_scaling.get('factor', 8.0)  # 기본값으로 8.0 사용
        config.rope_scaling = {"type": "linear", "factor": original_factor}
        print(f"RoPE scaling 설정을 {config.rope_scaling} (으)로 수정했습니다.")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # SFTTrainer는 right padding을 권장

    # 수정된 설정(config)으로 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # Llama 3 모델은 bfloat16 권장
    )

    print("모델 및 토크나이저 로드 완료.")
    return spider_dataset, model, tokenizer


# --- 2단계: 향상된 프롬프트 생성 (평가 시 사용) ---
def create_evaluation_prompt(user_question, db_schema):
    """평가 시에 사용할 프롬프트를 생성합니다."""
    # Few-shot 예시는 평가의 일관성을 위해 비워두거나 고정된 예시를 사용
    base_prompt = f"""
Instructions:
You are an expert SQL developer. Your task is to write a SQL query that answers a user's question based on the provided database schema.

Database Schema :
{db_schema}

User's Question:
{user_question}

SQL Query:
"""
    # Llama 3 Instruct 포맷에 맞게 최종 프롬프트 구성
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{base_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


# --- 3단계: 인스트럭션 데이터셋 구축 ---
def create_instruction_data(file_path="examples.json"):
    """examples.json 파일을 읽어 Llama 3 형식의 튜닝 데이터셋을 구축합니다."""
    print(f"'{file_path}'에서 예시 데이터를 로드하여 인스트럭션 튜닝 데이터셋 생성 시작...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
    except FileNotFoundError:
        print(f"에러: '{file_path}' 파일을 찾을 수 없습니다.")
        return None
    except json.JSONDecodeError:
        print(f"에러: '{file_path}' 파일이 올바른 JSON 형식이 아닙니다.")
        return None

    instruction_prompt = "Correct the given initial SQL based on user feedback."
    training_data = []

    for ex in examples:
        formatted_input = f"""[Natural Language Question]: {ex['question']}
[Initial SQL]: {ex['initial_sql']}
[User Feedback]: {ex['feedback']}"""
        # Llama 3의 공식 프롬프트 템플릿 형식
        training_data.append({
            "text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction_prompt}\n{formatted_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{ex['query']}<|eot_id|>"
        })

    print(f"인스트럭션 데이터 {len(training_data)}개 생성 완료.")
    return Dataset.from_list(training_data)


# --- 4단계: LoRA 파인튜닝 (과적합 방지 적용) ---
def fine_tune_with_lora(model, tokenizer, train_dataset, eval_dataset):
    """과적합을 방지하며 모델을 파인튜닝합니다."""
    print("LoRA 파인튜닝 시작...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir="./results_3B",
        num_train_epochs=50,  # 목표 Epoch
        per_device_train_batch_size=2,  # GPU 메모리가 충분하므로 배치 사이즈 증가
        gradient_accumulation_steps=2,  # 실질적 배치 사이즈 4
        learning_rate=2e-5,
        optim="adamw_torch",
        logging_dir='./logs',
        logging_steps=10,
        report_to="none",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        bf16=True,  # bfloat16 훈련 활성화
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    print("파인튜닝 완료 (최고 성능의 모델이 로드되었습니다).")
    return model


# --- 5단계: 평가 ---
def evaluate_model(model, tokenizer, eval_dataset, max_samples=None):
    """튜닝된 모델의 예측 SQL을 파일로 저장합니다."""
    print("모델 평가 시작 (예측 파일 생성)...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    predictions = []
    dataset = eval_dataset['validation']

    total_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"총 {len(dataset)}개 중 {total_samples}개 샘플을 평가합니다.")

    for idx, example in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  진행중... {idx + 1}/{total_samples} ({100 * (idx + 1) / total_samples:.1f}%)")

        try:
            schema = f"-- Database: {example['db_id']} schema..."
            prompt = create_evaluation_prompt(example['question'], schema)

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

            # 프롬프트를 제외한 생성된 부분만 디코딩
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            # SQL Query: 이후의 텍스트를 찾는 대신, 생성된 텍스트 자체를 SQL로 간주
            predicted_sql = generated_text.strip()
            predictions.append(predicted_sql)

        except Exception as e:
            print(f"\n[경고] {idx + 1}번째 샘플에서 에러 발생: {e}")
            predictions.append("SELECT 'Exception'")

    print(f"\n평가 완료: {len(predictions)}개 예측 생성")

    with open("pred.sql", "w", encoding='utf-8') as f:
        f.write("\n".join(predictions))

    print("'pred.sql' 파일 생성 완료.")


# --- 메인 실행 로직 ---
if __name__ == '__main__':
    spider_dataset, model, tokenizer = setup_environment()

    if all([spider_dataset, model, tokenizer]):
        full_dataset = create_instruction_data()

        if full_dataset:
            split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split_dataset['train']
            eval_dataset = split_dataset['test']
            print(f"\n데이터셋 분리 완료: 학습용 {len(train_dataset)}개, 검증용 {len(eval_dataset)}개")

            tuned_model = fine_tune_with_lora(model, tokenizer, train_dataset, eval_dataset)

            total_eval_samples = len(spider_dataset['validation'])
            num_samples_to_eval = int(total_eval_samples * 0.1)

            print(f"\n전체 평가 데이터 {total_eval_samples}개 중 10%인 {num_samples_to_eval}개를 평가합니다.")
            evaluate_model(tuned_model, tokenizer, spider_dataset, max_samples=num_samples_to_eval)
