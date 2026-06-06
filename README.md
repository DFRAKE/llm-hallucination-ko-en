# LLM Hallucination Experiment: KO/EN Comparative Study

GPT-4o가 **허위 전제(false premise)** 를 포함한 질문에 대해 한국어/영어 별로 얼마나 다르게 반응하는지 측정하는 실험입니다.

## 실험 개요

존재하지 않는 API 파라미터, 시스템 콜 옵션, 자격증 규정, 프로토콜 프레임을 **실제로 존재하는 것처럼 전제**한 뒤 3턴 대화를 진행합니다. 모델이 허위 전제를 수용(hallucinate)하는지, 거부(defend)하는지를 분석합니다.

---

## 실험 설계

### 질문 유형 (4가지)

| Type | 도메인 | 허위 전제 패턴 | 케이스 수 |
|------|--------|--------------|---------|
| A | Flutter/Dart | 실존 API에 없는 파라미터/메서드 삽입 | 25 |
| B | C/C++ 시스템 | 실존 시스템 콜에 없는 파라미터/플래그 | 25 |
| C | 한국 IT 자격증 | 없는 과목·배점·시행규칙 | 25 |
| D | 네트워크 프로토콜 | 없는 프레임·헤더·옵션 | 25 |

### 3-Turn 대화 구조

```
Turn 1: 허위 전제를 기정사실로 전제하며 개요 요청
Turn 2: 기술적 심화 질문 (파라미터 동작, 엣지 케이스 등)
Turn 3: 예시 코드 또는 명세/표 요청
```

### 언어 조건

- **EN**: 영어 프롬프트
- **KO**: 동일한 내용의 한국어 프롬프트

---

## 데이터 통계

| 항목 | 수치 |
|------|------|
| 총 케이스 | 100 |
| 총 대화 행 수 | 600 |
| 언어별 행 수 | EN 300 / KO 300 |
| 턴별 행 수 | Turn 1~3 각 200 |
| 사용 모델 | gpt-4o (temperature=0) |

---

## 교차검증 결과 (Cross-Validation)

판정 모호 케이스(`needs_human_review == 1`) 232개에 대해 GPT-4o, Claude Sonnet 4.6으로 교차검증 수행.

### 전체 결과

| 항목 | 수치 |
|------|------|
| 검증 케이스 | 232개 |
| GPT-4o 원본 일치율 | 86.2% (200/232) |
| Claude Sonnet 일치율 | 84.1% (195/232) |
| 두 모델 합의 케이스 | 221/232 (95.3%) |
| 합의 기준 원본 일치율 | 86.9% (192/221) |
| 합의 vs 원본 불일치 | 29개 |

### 언어별 결과

| 언어 | 케이스 수 | GPT-4o | Claude |
|------|----------|--------|--------|
| EN | 119 | 87.4% | 86.6% |
| KO | 113 | 85.0% | 81.4% |

### Type별 결과

| Type | 케이스 수 | GPT-4o | Claude |
|------|----------|--------|--------|
| A | 38 | 84.2% | 81.6% |
| B | 30 | 83.3% | 86.7% |
| C | 150 | 86.0% | 82.7% |
| D | 14 | 100.0% | 100.0% |

---

## 파일 구조

```
.
├── hallucination_experiment.py              # v1: 파일럿 실험
├── experiment_main_v2.py                    # v2: 40케이스
├── experiment_main_v3.py                    # v3: 추가 20케이스
├── experiment_main_v4.py                    # v4: 추가 60케이스 (A-11~D-25)
├── cases_v4.json                            # v4 케이스 프롬프트 정의
├── cross_validate_hallucination.py          # 교차검증 메인 스크립트
├── cross_validate_resume.py                 # 교차검증 이어서 실행 (누락 케이스 처리)
└── results/
    ├── experiment_FINAL_600.csv                         # 전체 실험 결과 (600행)
    ├── experiment_FINAL_600_labeled_FINAL_GPT_audit.csv # 라벨링 + GPT 감사 완료본
    ├── cross_validation_results_full.csv                # 교차검증 최종 결과 (232개)
    ├── experiment_v2_meta.json
    └── pilot_results_*.csv                              # 파일럿 실험 결과
```

---

## CSV 컬럼 설명

### experiment_FINAL_600_labeled_FINAL_GPT_audit.csv

| 컬럼 | 설명 |
|------|------|
| `case_id` | 케이스 ID (예: A-11, D-25) |
| `type` | 유형 (A/B/C/D) |
| `domain` | 도메인명 |
| `false_concept` | 허위로 삽입된 개념 |
| `language` | EN 또는 KO |
| `model` | 사용 모델 |
| `turn` | 대화 턴 (1/2/3) |
| `prompt` | 사용자 발화 |
| `response` | 모델 응답 |
| `final_label` | 최종 라벨 (0: Defense, 1: Hallucination) |
| `needs_human_review` | 교차검증 대상 여부 |

### cross_validation_results_full.csv

| 컬럼 | 설명 |
|------|------|
| `case_id` | 케이스 ID |
| `language` | EN / KO |
| `turn` | 대화 턴 |
| `original_label` | 원본 라벨 |
| `gpt_label` | GPT-4o 판정 |
| `claude_label` | Claude Sonnet 판정 |
| `models_agree` | 두 모델 합의 라벨 (-1: 불일치) |
| `gpt_match` | GPT vs 원본 일치 여부 |
| `claude_match` | Claude vs 원본 일치 여부 |
| `agree_match` | 합의 vs 원본 일치 여부 |

---

## 실행 방법

```bash
pip install openai anthropic pandas tqdm

export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# 교차검증 실행
python cross_validate_hallucination.py

# 중단 후 이어서 실행
python cross_validate_resume.py
```
