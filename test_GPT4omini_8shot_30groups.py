import os
import time

import openai
from tqdm import tqdm
import json
from extract_number import extract_last_number
from collections import defaultdict

# ---- Configurable parameters ----
num_templates = 50  # How many templates to evaluate
num_variants = 10  # How many variants for each template (index starts from 0)
# ---------------------

# File path
file_path = "Template_data/GSM_symbolic.jsonl"  # File path containing 50 groups of data
model_id = "gpt-4o-mini"
result_file = "accuracy_GPT4omini_8shot_10groups.jsonl"  # Output file
api_key = os.getenv("OPENAI_API_KEY")

# Model loading
client = openai.OpenAI(api_key=api_key)

messages_8shot = [
    {
        "role": "system",
        "content": "As an expert problem solver, solve step by step the following mathematical questions.",
    },
    {
        "role": "user",
        "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    },
    {
        "role": "assistant",
        "content": "Let’s think step by step. Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\nThe final answer is 72",
    },
    {
        "role": "user",
        "content": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
    },
    {
        "role": "assistant",
        "content": "Let’s think step by step. Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.\nThe final answer is 10",
    },
    {
        "role": "user",
        "content": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
    },
    {
        "role": "assistant",
        "content": "Let’s think step by step. In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nBetty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\nThe final answer is 5",
    },
    {
        "role": "user",
        "content": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
    },
    {
        "role": "assistant",
        "content": "Let’s think step by step. Julie read 12 x 2 = <<12*2=24>>24 pages today.\nSo she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.\nThere are 120 - 36 = <<120-36=84>>84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.\nThe final answer is 42",
    },
    {
        "role": "user",
        "content": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
    },
    {
        "role": "assistant",
        "content": "Let’s think step by step. He writes each friend 3*2=<<3*2=6>>6 pages a week.\nSo he writes 6*2=<<6*2=12>>12 pages every week.\nThat means he writes 12*52=<<12*52=624>>624 pages a year.\nThe final answer is 624",
    },
    {
        "role": "user",
        "content": "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of them are yellow, and there are 80% more of those in purple. There are only 25% as many green flowers as there are yellow and purple flowers. How many flowers does Mark have in his garden?",
    },
    {
        "role": "assistant",
        "content": "Let’s think step by step. There are 80/100 * 10 = <<80/100*10=8>>8 more purple flowers than yellow flowers.\nSo in Mark's garden, there are 10 + 8 = <<10+8=18>>18 purple flowers.\nPurple and yellow flowers sum up to 10 + 18 = <<10+18=28>>28 flowers.\nThat means in Mark's garden there are 25/100 * 28 = <<25/100*28=7>>7 green flowers.\nSo in total Mark has 28 + 7 = <<28+7=35>>35 plants in his garden.\nThe final answer is 35",
    },
    {
        "role": "user",
        "content": "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
    },
    {
        "role": "assistant",
        "content": "Let’s think step by step. He eats 32 from the largest pizzas because 2 x 16 = <<2*16=32>>32.\nHe eats 16 from the small pizza because 2 x 8 = <<2*8=16>>16.\nHe eats 48 pieces because 32 + 16 = <<32+16=48>>48.\nThe final answer is 48",
    },
    {
        "role": "user",
        "content": "Ken created a care package to send to his brother, who was away at boarding school. Ken placed a box on a scale, and then he poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly beans. And finally, he added enough gummy worms to double the weight once again. What was the final weight of the box of goodies, in pounds?",
    },
    {
        "role": "assistant",
        "content": "Let’s think step by step. To the initial 2 pounds of jelly beans, he added enough brownies to cause the weight to triple, bringing the weight to 2*3=<<2*3=6>>6 pounds.\nNext, he added another 2 pounds of jelly beans, bringing the weight to 6+2=<<6+2=8>>8 pounds.\nAnd finally, he added enough gummy worms to double the weight once again, to a final weight of 8*2=<<8*2=16>>16 pounds.\nThe final answer is 16",
    },
]

suffix = [{"role": "assistant", "content": "Let’s think step by step."}]

group_accuracies = []

# 1) Read all items and cluster by template id
all_items_by_id = defaultdict(list)
with open(file_path, "r", encoding="utf-8") as fin:
    for lineno, line in enumerate(fin, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception as e:
            print(f"[WARN] JSON decode error line {lineno}: {e}")
            continue
        all_items_by_id[int(item["id"])].append(item)

# For each selected template, sort variants by instance and truncate to num_variants
templates_variants = {}  # id -> list of variant items (ordered by instance)
for tid in range(num_templates):  # for tid in selected_template_ids:
    lst = all_items_by_id[tid]
    templates_variants[tid] = lst[:num_variants]
    print(
        f"Template id={tid}: found {len(lst)} variants, using {len(templates_variants[tid])}"
    )

# 2) Make one model call for each (template, variant_idx) and record if the prediction is correct at that position
# results_by_variant_idx[v] will be list of booleans (True=correct, False=incorrect) for each template that had that variant
results_by_variant_idx = defaultdict(list)

print("\nStarting model calls for selected templates/variants...")
total_calls = num_templates * num_variants
pbar = tqdm(total=total_calls, desc="GPT calls", unit="call")
for tid in range(num_templates):  # for tid in selected_template_ids:
    variants = templates_variants.get(tid, [])
    for v_idx in range(num_variants):
        # if this template doesn't have this variant index (shorter), skip
        if v_idx >= len(variants):
            # no sample for this template at this variant index
            continue
        data = variants[v_idx]
        question = data["question"]
        corr_ans = extract_last_number(data["answer"])

        # construct messages
        messages_for_call = (
            messages_8shot + [{"role": "user", "content": question}] + suffix
        )

        completion = None
        try:
            completion = client.chat.completions.create(
                model=model_id, messages=messages_for_call, temperature=0, top_p=1
            )
        except Exception as e:
            if "rate_limit" in str(e):
                time.sleep(0.5)
            else:
                print(f"Error in GPT request: {e}")
                continue  # Skip error samples

        if completion is None:
            # we don't append anything to results_by_variant_idx for API-failed sample (it won't contribute to denom)
            continue

        # parse model response
        try:
            model_text = completion.choices[0].message.content
            pred_ans = extract_last_number(model_text)
            is_correct = abs(pred_ans - corr_ans) < 1e-4
            results_by_variant_idx[v_idx].append(bool(is_correct))
        except Exception as e:
            print(f"[WARN] parse/other error for template {tid} variant {v_idx}: {e}")
            continue
        pbar.update(1)

# 3) Calculate group accuracy by variant_idx (0..num_variants-1) and write to result_file
with open(result_file, "w", encoding="utf-8") as write_file:
    group_accuracies = []
    for v_idx in range(num_variants):
        per_list = results_by_variant_idx.get(v_idx, [])
        total_questions = len(per_list)  # number of templates that produced a valid judged result at this variant index
        corr_preds = sum(1 for x in per_list if x is True)

        accuracy = corr_preds / total_questions if total_questions > 0 else 0.0

        group_result = {
            "group_index": v_idx + 1,
            "accuracy": accuracy,
            "total_questions": total_questions,
            "correct_predictions": corr_preds,
        }
        write_file.write(json.dumps(group_result, ensure_ascii=False) + "\n")
        group_accuracies.append(group_result)

# 4) Print a summary to the console
print("\nVariant-group summary (first 10):")
for gr in group_accuracies[:10]:
    print(
        f"Group {gr['group_index']}: acc={gr['accuracy']:.4f} ({gr['correct_predictions']}/{gr['total_questions']})"
    )

print(f"\nSaved variant-group results to: {result_file}")
