# LLM Hallucination Experiment: KO/EN Comparative Study

GPT-4o가 **허위 전제(false premise)** 를 포함한 질문에 대해 한국어/영어 별로 얼마나 다르게 반응하는지 측정하는 실험입니다.

## 실험 개요

존재하지 않는 API 파라미터, 시스템 콜 옵션, 자격증 규정, 프로토콜 프레임을 **실제로 존재하는 것처럼 전제**한 뒤 3턴 대화를 진행합니다. 모델이 허위 전제를 수용(hallucinate)하는지, 거부(defend)하는지를 분석합니다.

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
         (Type D: 명세·의사코드만, 공격 코드 금지)
         (Type C: 표 형태)
```

### 언어 조건

- **EN**: 영어 프롬프트
- **KO**: 동일한 내용의 한국어 프롬프트

## 데이터 통계

| 항목 | 수치 |
|------|------|
| 총 케이스 | 100 |
| 총 대화 행 수 | 600 |
| 언어별 행 수 | EN 300 / KO 300 |
| 턴별 행 수 | Turn 1~3 각 200 |
| 사용 모델 | gpt-4o (temperature=0) |

## 파일 구조

```
.
├── hallucination_experiment.py          # v1: 파일럿 실험
├── experiment_main_v2.py                # v2: 40케이스 (A-01~D-10)
├── experiment_main_v3.py                # v3: 추가 20케이스
├── experiment_main_v4.py                # v4: 추가 60케이스 (A-11~D-25)
├── cases_v4.json                        # v4 케이스 프롬프트 정의
└── results/
    ├── experiment_FINAL_600.csv         # 전체 합산 결과 (100케이스 × 2언어 × 3턴)
    ├── experiment_COMBINED.csv          # v1~v3 합산
    ├── experiment_v4_additional_results_*.csv  # v4 개별 실행 결과
    ├── experiment_v2_results_FINAL_*.csv
    ├── experiment_v3_additional_results_*.csv
    └── pilot_results_*.csv              # 파일럿 결과
```

## CSV 컬럼 설명

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
| `defense_cue` | 방어 단서 (레이블링용, 수동 입력) |
| `hallucination` | 할루시네이션 여부 (레이블링용) |
| `judgment_note` | 판정 메모 (레이블링용) |

## 실행 방법

```bash
# 의존성 설치
pip install openai pandas

# API 키 설정
export OPENAI_API_KEY="your-key-here"

# v4 실험 실행 (이미 완료된 케이스 자동 스킵)
python3 experiment_main_v4.py
```

## 주요 관찰 (예비)

- **Type D (네트워크 프로토콜)**: Turn 1부터 "This is not a standard option"으로 방어하는 경향
- **Type C (자격증)**: "indeed been restructured"처럼 허위 전제를 수용하는 경향
- EN/KO 언어 간 방어율 차이는 레이블링 후 정량 분석 예정
