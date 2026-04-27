import re
import argparse
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer
from bert_score import score as bertscore_score

OUTPUT_COLUMNS = [
    "baseline_output",
    "finetuned_output",
    "advanced_output",
    "debate_output",
    "dpo_output",
    "dpo_advanced_output",
]

SLANG_LEXICON = {
    "u", "ur", "urs", "r", "ya", "yall", "ya'll", "imma", "gonna", "wanna", "gotta",
    "ain't", "aint", "idk", "ikr", "btw", "tbh", "brb", "lol", "lmao", "lmfao", "rofl",
    "wtf", "omg", "tho", "tho.", "cuz", "coz", "bc", "b/c", "plz", "pls", "thx", "ty",
    "kinda", "sorta", "lemme", "gimme", "tryna", "gotchu", "dunno", "wassup", "sup",
    "nah", "yep", "bro", "bruh", "sis", "homie", "wanna", "hafta", "outta", "coulda",
    "woulda", "shoulda", "finna", "gon", "yo", "asap", "nvm", "imo", "imho", "irl",
    "dm", "pm", "bff", "tmi", "fyi", "smh", "fr", "ngl", "lowkey", "highkey", "cap",
    "nocap", "tho", "cus", "luv", "cya", "tho", "alr", "rn", "bcuz", "yaa", "thooo"
}


def tokenize(text):
    if pd.isna(text):
        return []
    return re.findall(r"[A-Za-z']+", str(text).lower())


def residual_slang_rate(original_text, output_text, slang_lexicon):
    original_tokens = set(tokenize(original_text))
    output_tokens = set(tokenize(output_text))
    slang_in_original = {t for t in original_tokens if t in slang_lexicon}
    if not slang_in_original:
        return 0.0
    unresolved = slang_in_original.intersection(output_tokens)
    return len(unresolved) / len(slang_in_original)


def residual_slang_count(output_text, slang_lexicon):
    output_tokens = tokenize(output_text)
    return sum(1 for t in output_tokens if t in slang_lexicon)


def evaluate_system(df, pred_col, ref_col="ground_truth", source_col="original_slang"):
    refs = df[ref_col].fillna("").astype(str).tolist()
    preds = df[pred_col].fillna("").astype(str).tolist()
    sources = df[source_col].fillna("").astype(str).tolist()

    bleu = BLEU(effective_order=True)
    chrf = CHRF(word_order=2)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    bleu_score = bleu.corpus_score(preds, [refs]).score
    chrf_score = chrf.corpus_score(preds, [refs]).score

    rouge_l_scores = []
    for ref, pred in zip(refs, preds):
        rouge_l_scores.append(rouge.score(ref, pred)["rougeL"].fmeasure)
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    p, r, f1 = bertscore_score(
        preds,
        refs,
        lang="en",
        verbose=False,
        rescale_with_baseline=True
    )
    bert_f1 = f1.mean().item()

    residual_rates = [
        residual_slang_rate(src, pred, SLANG_LEXICON)
        for src, pred in zip(sources, preds)
    ]
    avg_residual_slang_rate = sum(residual_rates) / len(residual_rates)

    residual_counts = [
        residual_slang_count(pred, SLANG_LEXICON)
        for pred in preds
    ]
    avg_residual_slang_count = sum(residual_counts) / len(residual_counts)

    return {
        "system": pred_col,
        "BLEU": round(bleu_score, 4),
        "ROUGE_L": round(rouge_l, 4),
        "chrF": round(chrf_score, 4),
        "BERTScore_F1": round(bert_f1, 4),
        "Residual_Slang_Rate": round(avg_residual_slang_rate, 4),
        "Avg_Residual_Slang_Count": round(avg_residual_slang_count, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to combined evaluation CSV")
    parser.add_argument("--output", type=str, default="evaluation_summary.csv", help="Path to save summary CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    missing = [c for c in ["original_slang", "ground_truth"] + OUTPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    results = []
    for col in OUTPUT_COLUMNS:
        results.append(evaluate_system(df, col))

    summary = pd.DataFrame(results)
    summary = summary.sort_values(
        by=["BERTScore_F1", "chrF", "BLEU"],
        ascending=[False, False, False]
    )

    print(summary.to_string(index=False))
    summary.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
