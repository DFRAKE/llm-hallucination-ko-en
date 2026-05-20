import os
import json
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from openai import OpenAI

MODEL = "gpt-4o"
TEMPERATURE = 0
MAX_TOKENS = 800
SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's questions accurately and concisely."
RESULTS_DIR = Path("./results")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def load_cases():
    with open("cases_v4.json", "r", encoding="utf-8") as f:
        return json.load(f)

def call_api(messages):
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=messages,
    )
    return resp.choices[0].message.content

def run_case(case_id, case):
    rows = []
    for lang in ["EN", "KO"]:
        turns = case[lang]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for i, prompt in enumerate(turns):
            messages.append({"role": "user", "content": prompt})
            response = call_api(messages)
            messages.append({"role": "assistant", "content": response})
            rows.append({
                "case_id": case_id,
                "type": case["type"],
                "domain": case["domain"],
                "false_concept": case["false_concept"],
                "language": lang,
                "model": MODEL,
                "turn": i + 1,
                "prompt": prompt,
                "response": response,
                "defense_cue": "",
                "hallucination": "",
                "judgment_note": "",
            })
            time.sleep(2)
    return rows

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    cases = load_cases()

    # 완료된 케이스 확인
    temp_path = RESULTS_DIR / "experiment_v4_temp.csv"
    completed = set()
    all_rows = []
    if temp_path.exists():
        df_temp = pd.read_csv(temp_path)
        completed = set(df_temp["case_id"].unique())
        all_rows = df_temp.to_dict("records")
        print(f"이미 완료된 케이스: {sorted(completed)} ({len(completed)}개)")

    remaining = [(cid, c) for cid, c in cases.items() if cid not in completed]
    print(f"남은 케이스: {len(remaining)}개")

    for idx, (case_id, case) in enumerate(remaining):
        print(f"[{idx+1}/{len(remaining)}] {case_id} 실행 중...")
        try:
            rows = run_case(case_id, case)
            all_rows.extend(rows)
            df_temp = pd.DataFrame(all_rows)
            df_temp.to_csv(temp_path, index=False, encoding="utf-8-sig")
            print(f"  -> {case_id} 완료, 중간저장 완료")
        except Exception as e:
            print(f"  -> {case_id} 오류: {e}")
            time.sleep(10)
            continue

    # 최종 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = RESULTS_DIR / f"experiment_v4_additional_results_{timestamp}.csv"
    df_final = pd.DataFrame(all_rows)
    df_final.to_csv(final_path, index=False, encoding="utf-8-sig")
    print(f"\n최종 저장: {final_path}")
    print(f"총 수집 행 수: {len(df_final)}")

    # 기존 데이터와 합산
    combined_path = RESULTS_DIR / "experiment_COMBINED.csv"
    if combined_path.exists():
        existing = pd.read_csv(combined_path)
        combined = pd.concat([existing, df_final], ignore_index=True)
        combined = combined.drop_duplicates(subset=["case_id", "language", "turn"])
        combined.to_csv(RESULTS_DIR / "experiment_FINAL_600.csv", index=False, encoding="utf-8-sig")
        print(f"기존: {len(existing)}행 | 신규: {len(df_final)}행 | 합산: {len(combined)}행")

if __name__ == "__main__":
    main()
