import os
# bitsandbytes를 사용하지 않도록 환경 변수 설정
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import re
import json


# --- 1단계: 모델 및 데이터셋 로드 ---
def setup_environment():
    """모델, 토크나이저, 데이터셋을 로드하고 설정합니다."""
    print("환경 설정 시작...")
    spider_dataset = load_dataset("spider")
    print("Spider 데이터셋 로드 완료.")
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 생성을 위해 왼쪽 패딩 사용
    
    # 모델 로드 
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto", 
        trust_remote_code=True,
        torch_dtype=torch.float16, 
        attn_implementation="eager"  
    )
    print("모델 및 토크나이저 로드 완료.")
    return spider_dataset, model, tokenizer



# --- 2단계: 향상된 프롬프트 생성 ---
def create_enhanced_prompt(user_question, db_schema):
    """논문의 부록 A를 기반으로 구조화된 프롬프트를 생성합니다."""
    prompt_template = f"""
Instructions:
You are an expert SQL developer. Your task is to write a SQL query that answers a user's question based on the provided database schema.
Think step-by-step to arrive at the correct query. First, analyze the question to identify the necessary tables and columns.
Then, determine the required JOIN conditions, aggregations, and filtering clauses. Finally, construct the complete SQL query.

Database Schema :
{db_schema}

Few-shot Examples:
-- Example 1 (유사한 질문/SQL 쌍으로 채워야 함)
-- Question: ...
SELECT ...

-- Example 2 (유사한 질문/SQL 쌍으로 채워야 함)
-- Question: ...
SELECT ...

User's Question:
{user_question}

SQL Query:
"""
    return prompt_template

# --- 3단계: 피드백 시뮬레이션 및 인스트럭션 데이터셋 구축 ---
def simulate_feedback_and_create_instruction_data(file_path="examples.json"):
    """
    examples.json 파일에서 예시를 읽어와 
    인스트럭션 튜닝 데이터셋을 구축합니다.
    """
    print(f"'{file_path}'에서 예시 데이터를 로드하여 인스트럭션 튜닝 데이터셋 생성 시작...")
    
    # JSON 파일 읽기
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
    except FileNotFoundError:
        print(f"에러: '{file_path}' 파일을 찾을 수 없습니다. 스크립트와 같은 디렉토리에 파일을 생성해주세요.")
        return []
    except json.JSONDecodeError:
        print(f"에러: '{file_path}' 파일이 올바른 JSON 형식이 아닙니다.")
        return []

    instruction_prompt = "Correct the given initial SQL based on user feedback."
    training_data = []
    
    for ex in examples:
        # 논문 Figure 2의 데이터 구조로 변환
        formatted_input = f"""[Natural Language Question]: {ex['question']}
[Initial SQL]: {ex['initial_sql']}
[User Feedback]: {ex['feedback']}"""

        # 최종 학습 데이터 포맷
        training_data.append({
            "text": f"<s>[INST] {instruction_prompt}\n{formatted_input} [/INST] {ex['query']} </s>"
        })
        
    print(f"인스트럭션 데이터 {len(training_data)}개 생성 완료.")
    return training_data
# --- 4단계: LoRA를 이용한 파인튜닝 ---
def fine_tune_with_lora(model, tokenizer, training_data):
    """LoRA를 사용하여 모델을 파인튜닝합니다."""
    print("LoRA 파인튜닝 시작...")
    
    # 모델의 pad_token_id 명시적 설정
    model.config.pad_token_id = tokenizer.pad_token_id
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        optim="adamw_torch",  # optimizer -> optim으로 변경
        logging_dir='./logs',
        logging_steps=10,
        do_eval=False,
        report_to="none",
        max_grad_norm=1.0,  # gradient clipping으로 안정성 향상
        fp16=False,  # float16 연산 비활성화 (더 안정적)
        dataloader_pin_memory=False,  # 메모리 관련 이슈 방지
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=training_data,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()
    print("파인튜닝 완료.")
    return model

# --- 5단계: 평가 (공식 스크립트 사용을 위해 수정) ---
def evaluate_model(model, tokenizer, eval_dataset, max_samples=None):
    """튜닝된 모델의 예측 SQL을 파일로 저장합니다.
    
    Args:
        model: 평가할 모델
        tokenizer: 토크나이저
        eval_dataset: 평가 데이터셋
        max_samples: 평가할 최대 샘플 수 (None이면 전체)
    """
    print("모델 평가 시작 (예측 파일 생성)...")
    
    # CUDA 캐시 클리어 (메모리 정리)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA 캐시 클리어 완료")
    
    predictions = []
    split_name = 'validation'
    dataset = eval_dataset[split_name]
    
    # 전체 개수 확인
    total_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    print(f"총 {len(dataset)}개 중 {total_samples}개 샘플을 평가합니다.")
    print("진행 상황: (이 작업은 시간이 오래 걸릴 수 있습니다)")
    
    # 평가 데이터셋을 순회하며 예측 생성
    for idx, example in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
            
        # 진행 상황 출력 (10개마다)
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  진행중... {idx + 1}/{total_samples} ({100*(idx+1)/total_samples:.1f}%)")
        
        try:
            question = example['question']
            db_id = example['db_id']
            # [중요] 실제 스키마 정보가 필요하지만, 간소화를 위해 프롬프트는 그대로 둡니다.
            schema = f"-- Database: {db_id} schema information goes here."
            prompt = create_enhanced_prompt(question, schema)
            
            device = next(model.parameters()).device
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            sql_match = re.search(r'SQL Query:\s*(.*)', generated_text, re.DOTALL)
            
            if sql_match:
                predicted_sql = sql_match.group(1).strip()
            else:
                predicted_sql = "SELECT 'Error'" # 생성 실패 시
                
            predictions.append(predicted_sql)
            
        except Exception as e:
            print(f"\n[경고] {idx+1}번째 샘플에서 에러 발생: {e}")
            predictions.append("SELECT 'Exception'")
            continue

    print(f"\n평가 완료: {len(predictions)}개 예측 생성")
    
    # 예측 결과를 pred.sql 파일에 저장
    with open("pred.sql", "w", encoding='utf-8') as f:
        for sql in predictions:
            f.write(sql + "\n")
            
    print("'pred.sql' 파일 생성 완료.")
    print("이제 터미널에서 Spider 공식 평가 스크립트를 실행하세요.")

# --- 메인 실행 로직 ---
if __name__ == '__main__':
    spider_dataset, model, tokenizer = setup_environment()
    
    # 3단계: 인스트럭션 데이터 생성
    from datasets import Dataset
    # 함수 인자 제거
    instruction_data_list = simulate_feedback_and_create_instruction_data()
    instruction_dataset = Dataset.from_list(instruction_data_list)

    # 생성된 데이터 확인
    print("\n--- 생성된 첫 번째 학습 데이터 예시 ---")
    print(json.dumps(instruction_dataset[0], indent=2, ensure_ascii=False))
    
    # 4단계: 모델 파인튜닝
    tuned_model = fine_tune_with_lora(model, tokenizer, instruction_dataset)
    
    # 5단계: 평가
    evaluate_model(tuned_model, tokenizer, spider_dataset, max_samples=50)
