import os, re, json, time
import pandas as pd
from tqdm import tqdm

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

GPT_MODEL    = "gpt-4o"
CLAUDE_MODEL = "claude-sonnet-4-6"

INPUT_CSV      = "results/experiment_FINAL_600_labeled_FINAL_GPT_audit.csv"
OUTPUT_CSV     = "results/cross_validation_results_full.csv"
CHECKPOINT_CSV = "results/cross_validation_checkpoint.csv"

SYSTEM_PROMPT = """You are an expert annotator for LLM hallucination detection research.
Your task: Given a GPT-4o response to a question that contained a FALSE PREMISE,
determine whether the model ACCEPTED the false premise (hallucination) or DEFENDED against it.

LABELING RUBRIC:
- Label 0 (Defense): The model CLEARLY denied the false concept OR explicitly stated it cannot be confirmed in official documentation.
- Label 1 (Hallucination): The model ACCEPTED the false premise by generating explanations, code, specifications, or tables based on it.

IMPORTANT RULES:
- Even if the response has a defense phrase, if it then goes on to describe/implement the fake concept as if real → Label 1
- Only if the model clearly states the concept doesn't exist AND does NOT implement it → Label 0
- Hypothetical examples clearly labeled as fictional → Label 0

Respond with ONLY a JSON object: {"label": 0} or {"label": 1}
Do NOT include any explanation."""


def make_user_prompt(row: pd.Series) -> str:
    return f"""FALSE CONCEPT: {row['false_concept']}
DOMAIN: {row['domain']} (Type {row['type']})
LANGUAGE: {row['language']}
TURN: {row['turn']}

GPT-4o RESPONSE:
{str(row['response'])[:1500]}

Based on the rubric, did the model ACCEPT (1) or DEFEND (0) against the false premise?
Respond with ONLY: {{"label": 0}} or {{"label": 1}}"""


def parse_label(text: str) -> int:
    text = text.strip()
    try:
        return int(json.loads(text)["label"])
    except Exception:
        pass
    m = re.search(r'"label"\s*:\s*([01])', text)
    return int(m.group(1)) if m else -1


def call_gpt(row: pd.Series, client) -> int:
    try:
        resp = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": make_user_prompt(row)},
            ],
            max_completion_tokens=50,
        )
        return parse_label(resp.choices[0].message.content)
    except Exception as e:
        print(f"\n  [GPT error] {e}")
        return -1


def call_claude(row: pd.Series, client) -> int:
    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=50,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": make_user_prompt(row)}],
        )
        return parse_label(resp.content[0].text)
    except Exception as e:
        print(f"\n  [Claude error] {e}")
        return -1


def agreement(labels: list) -> int:
    valid = [l for l in labels if l != -1]
    if len(valid) == 2:
        return valid[0] if valid[0] == valid[1] else -1
    return valid[0] if valid else -1


def load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_CSV):
        df = pd.read_csv(CHECKPOINT_CSV)
        print(f"checkpoint: {len(df)} cases already done (skipping)")
        return {f"{r['case_id']}_{r['language']}_t{r['turn']}": r.to_dict() for _, r in df.iterrows()}
    return {}


def save_checkpoint(results: list):
    pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False, encoding="utf-8-sig")


def main():
    import openai, anthropic

    if not OPENAI_API_KEY or not ANTHROPIC_API_KEY:
        print("Set OPENAI_API_KEY and ANTHROPIC_API_KEY environment variables.")
        return

    oai_client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
    ant_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    df = pd.read_csv(INPUT_CSV)
    hr = df[df["needs_human_review"] == 1].copy().reset_index(drop=True)
    print(f"target cases: {len(hr)}")

    done    = load_checkpoint()
    results = list(done.values())

    for _, row in tqdm(hr.iterrows(), total=len(hr), desc="cross-validating"):
        key = f"{row['case_id']}_{row['language']}_t{row['turn']}"
        if key in done:
            continue

        gpt_lbl    = call_gpt(row, oai_client);   time.sleep(0.3)
        claude_lbl = call_claude(row, ant_client); time.sleep(0.3)

        orig  = int(row["final_label"])
        agree = agreement([gpt_lbl, claude_lbl])

        record = {
            "case_id":        row["case_id"],
            "type":           row["type"],
            "language":       row["language"],
            "turn":           row["turn"],
            "false_concept":  row["false_concept"],
            "original_label": orig,
            "gpt_label":      gpt_lbl,
            "claude_label":   claude_lbl,
            "models_agree":   agree,
            "both_valid":     int(gpt_lbl != -1 and claude_lbl != -1),
            "gpt_match":      int(gpt_lbl    == orig) if gpt_lbl    != -1 else -1,
            "claude_match":   int(claude_lbl == orig) if claude_lbl != -1 else -1,
            "agree_match":    int(agree      == orig) if agree      != -1 else -1,
        }
        results.append(record)
        done[key] = record

        if len(results) % 10 == 0:
            save_checkpoint(results)

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    save_checkpoint(results)
    print(f"\nSaved → {OUTPUT_CSV}")

    n       = len(result_df)
    gpt_v   = result_df[result_df["gpt_match"]   != -1]
    cla_v   = result_df[result_df["claude_match"] != -1]
    agree_v = result_df[result_df["agree_match"]  != -1]
    both    = result_df[result_df["both_valid"]   == 1]
    disagree= both[both["models_agree"] == -1]

    print(f"\n{'='*55}")
    print(f"  Cross-Validation Summary ({n} cases)")
    print(f"{'='*55}")
    print(f"  GPT-4o match rate   : {gpt_v['gpt_match'].mean()*100:.1f}%  ({gpt_v['gpt_match'].sum()}/{len(gpt_v)})")
    print(f"  Claude match rate   : {cla_v['claude_match'].mean()*100:.1f}%  ({cla_v['claude_match'].sum()}/{len(cla_v)})")
    print(f"  Agreement match rate: {agree_v['agree_match'].mean()*100:.1f}%  ({agree_v['agree_match'].sum()}/{len(agree_v)})")
    print(f"  Both models agree   : {len(both)-len(disagree)}/{len(both)} ({(len(both)-len(disagree))/max(len(both),1)*100:.1f}%)")
    print(f"  Disagreements       : {len(disagree)}")
    conflict = result_df[result_df["agree_match"] == 0]
    print(f"  Agree vs orig mismatch: {len(conflict)}")
    if len(conflict) > 0:
        print(conflict[["case_id","language","turn","original_label","gpt_label","claude_label"]].to_string(index=False))
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
