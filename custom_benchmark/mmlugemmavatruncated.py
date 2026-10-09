

#pip install -q -U transformers datasets accelerate bitsandbytes

import numpy as np
import sys,os, json, csv, random, time, gc
from google.colab import userdata
import collections
import torch
import transformers
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,AutoConfig, AutoModelForCausalLM
from datasets import get_dataset_config_names, load_dataset

SYSTEM = "You must output the Answer"

# Build the Sinhala prompt for the model
def build_user(question, choices, subject_original=None, intro=False):
    n = len(choices)
    nums = ", ".join(str(i + 1) for i in range(n))
    body = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(choices))
    s = ""
    if intro and subject_original:
        s += f"මෙය {subject_original} විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි.\n"
    s += f"පහත ප්‍රශ්නයට {nums} යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරන්න.\n\n"
    s += f"ප්‍රශ්නය: {question}\n{body}\nපිළිතුර:"
    return s

# funtion to apply the model's chat template
def chat_text(tok, user):
    """Apply the model's chat template. Returns (text, used_template)."""
    candidates = (
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        [{"role": "user", "content": SYSTEM + "\n\n" + user}],  # templates without a system role
    )
    for msgs in candidates:
        try:
            text = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            return text, True
        except Exception:
            continue
    return user, False  # base model without a chat template

#Count parameters without downloading the weights
def count_params(name):

    """Parameter count from an empty (meta-device) model: no weights downloaded."""

    cfg = AutoConfig.from_pretrained(name)
    classes = [("AutoModelForCausalLM", AutoModelForCausalLM),
               ("AutoModelForMultimodalLM", getattr(transformers, "AutoModelForMultimodalLM", None))]
    out = {}
    for label, cls in classes:
        if cls is None:
            continue
        try:
            with init_empty_weights():
                m = cls.from_config(cfg)
            out[label] = sum(p.numel() for p in m.parameters())
        except Exception as e:  # noqa
            out[label] = f"n/a ({type(e).__name__})"
    return out

""" funtion to load only the language model of Gemma4_8B"""
def load_gemma4_8B(name, quant="4bit"):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this loader.")

    # Choose compute dtype based on GPU capability
    cc = torch.cuda.get_device_capability()[0]
    dtype = torch.bfloat16 if cc >= 8 else torch.float16

    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(name)

    # Load ONLY the text configuration
    print("Loading text configuration...")
    full_config = AutoConfig.from_pretrained(name)
    text_config = full_config.text_config

    kw = {"torch_dtype": dtype}

    if quant == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
    elif quant != "none":
        raise ValueError("quant must be '4bit' or 'none'")

    print("Loading text-only Gemma 4 model...")

    # Don't swallow the real loading exception
    model, info = transformers.Gemma4ForCausalLM.from_pretrained(
        name,
        config=text_config,
        device_map="cuda:0",
        output_loading_info=True,
        key_mapping={
            r"^model\.language_model\.": "model."
        },
        **kw,
    )

    model.eval()

    missing = sorted(info["missing_keys"])
    unexpected = sorted(info["unexpected_keys"])

    print("\nLoading report")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if missing:
        print("First missing keys:", missing[:10])

    if unexpected:
        print("First unexpected keys:", unexpected[:10])

    # Inspect the instantiated model
    has_vision = any(
        "vision" in n.lower()
        for n, _ in model.named_parameters()
    )
    has_audio = any(
        "audio" in n.lower()
        for n, _ in model.named_parameters()
    )

    total = sum(p.numel() for p in model.parameters())

    print("\nModel verification")
    print("Vision parameters present:", has_vision)
    print("Audio parameters present:", has_audio)
    print(f"Total logical parameters: {total:,}")
    print(f"Total logical parameters: {total / 1e9:.4f}B")

    return model, tok

def digit_ids_for(tok, k=5):

    ids = []
    for i in range(1, k + 1):
        enc = tok.encode(str(i), add_special_tokens=False)
        if len(enc) != 1:
            print(f"WARNING: digit {i} encodes to {len(enc)} tokens: {enc}")
        ids.append(enc[-1])
    print("digit token ids:", ids, [tok.decode([i]) for i in ids])
    return ids

# Run one forward pass and get the logits then apply the probabilities
def digit_logprobs(model, tok, user, n, digit_ids):

    """Log-softmax over the n digit logits at the last position. Returns (array, n_tokens)."""

    text, has_tpl = chat_text(tok, user)
    inputs = tok(text, return_tensors="pt", add_special_tokens=not has_tpl).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1].float()
    sel = logits[torch.tensor(digit_ids[:n], device=logits.device)]
    return torch.log_softmax(sel, dim=-1).cpu().numpy(), int(inputs["input_ids"].shape[1])

# return fraction as a percentage
def pct(x):
    return f"{100 * x:5.1f}%"

# computes accuracy grouped by any key (difficulty, category, subject, option count), optionally ignoring groups with fewer than min_n items.
def acc_by(results, pred_key, group_fn, min_n=1):
    g = collections.defaultdict(lambda: [0, 0])
    for r in results:
        k = group_fn(r)
        g[k][1] += 1
        g[k][0] += int(r[pred_key] == r["gold"])
    return {k: (c / t, t) for k, (c, t) in g.items() if t >= min_n}

# print reports
"""
prints overall accuracy, the breakdowns, the 5 weakest and 5 strongest subjects (with at least 15 questions),
and a slot histogram comparing how often the model predicted each position with how often each position is actually correct. That histogram reveals position bias, such as a model that loves answering "1".

"""

def print_report(results, pred_key, title):
    tot = np.mean([r[pred_key] == r["gold"] for r in results])
    print(f"\n=== {title}: overall {pct(tot)} (n={len(results)}) ===")
    for label, fn, mn in (("difficulty", lambda r: r["difficulty"], 1),
                          ("category", lambda r: r["category"], 1),
                          ("options", lambda r: len(r["choices"]), 1)):
        parts = [f"{k}: {pct(a)} (n={t})" for k, (a, t) in sorted(acc_by(results, pred_key, fn, mn).items(), key=str)]
        print(f"  by {label:10s} " + " | ".join(parts))
    subj = sorted(acc_by(results, pred_key, lambda r: r["subject"], 15).items(), key=lambda kv: kv[1][0])
    if subj:
        print("  weakest subjects (n>=15): " + ", ".join(f"{k} {pct(a)}" for k, (a, _) in subj[:5]))
        print("  strongest subjects:       " + ", ".join(f"{k} {pct(a)}" for k, (a, _) in subj[-5:]))
    pc = np.bincount([r[pred_key] for r in results], minlength=5)
    gc = np.bincount([r["gold"] for r in results], minlength=5)
    print(f"  predicted slot counts {pc.tolist()}   gold slot counts {gc.tolist()}")

""" Load a Hub dataset using load_dataset() """
def load_hf(name, config, split):

    token = userdata.get('HF_TOKEN')
    if not token:
        print("NOTE: HF_TOKEN is not set; gated datasets will fail to load.")
    try:
        """" CohereLabs/Global-MMLU", "si", split="test  if the dataset has configs"""
        ds = load_dataset(name, config, token=token) if config else load_dataset(name, token=token)
    except Exception as e:
        print(f"load_dataset failed: {type(e).__name__}: {e}")
        try:
            print("available configs:", get_dataset_config_names(name, token=token))
        except Exception:
            pass
        sys.exit("Fix the error above (token / accepted dataset terms / --config) and retry.")

    if not hasattr(ds, "keys"):  # a single Dataset
        ds = {"all": ds}
    print("\n=== splits ===")

    for k, v in ds.items():
        print(f"  {k}: {len(v)} rows, columns={v.column_names}")
    if split not in ds:
        sys.exit(f"Choose a split with --split (available: {list(ds.keys())}). "
                 "Use the one the organisers designate as the dev set.")
    return [dict(r) for r in ds[split]]

"""Decide whether answers are 0-based or 1-based for the whole dataset."""
def resolve_answer_index(recs, forced=None):
    vals = [r["answer"] for r in recs if r["answer"] is not None]
    max_n = max(len(r["choices"]) for r in recs)

    if not vals:
        base, why = 1, "no answers found"
    elif forced in (0, 1):
        base, why = forced, "forced by --answer_base"
    elif min(vals) == 0 and max(vals) == max_n:
        base, why = 0, f"INCONSISTENT: has both 0 and {max_n}"
    elif min(vals) == 0:
        base, why = 0, "found answer 0 -> 0-based"
    elif max(vals) == max_n:
        base, why = 1, f"found answer {max_n} -> 1-based"
    else:
        base, why = 1, "AMBIGUOUS (defaulting to 1-based; check the examples)"

    for r in recs:
        r["base"] = base
        r["gold"] = None if r["answer"] is None else r["answer"] - base

    print(f"\n=== ANSWER-BASE CHECK ===")
    print(f"questions={len(recs)}  answer counts={sorted(collections.Counter(vals).items())}")
    print(f"decision: {base}-based  [{why}]")
    shown = 0
    for r in recs:
        if r["gold"] is not None and 0 <= r["gold"] < len(r["choices"]) and shown < 3:
            print(f"  Q: {r['question'][:70]}")
            print(f"     raw answer={r['answer']} -> option text: {r['choices'][r['gold']][:60]}")
            shown += 1
    print("Confirm the option text is really the correct answer; otherwise rerun with --answer_base 0 or 1.\n")

def to_int(a):
    """Answer -> int. Accepts 3, '3', or a letter A-E (mapped to 1-5)."""
    if a is None:
        return None
    s = str(a).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    if len(s) == 1 and s.upper() in "ABCDE":
        return ord(s.upper()) - 64
    return None

def normalize(r, src):
    """Turn one raw dataset row into the flat record the rest of the code expects."""
    md = r.get("metadata") or {}
    return {
        "src": src,
        "q_no": r.get("q_no"),
        "subject": r.get("subject") or md.get("subject") or "unknown",
        "category": r.get("category") or "unknown",
        "subject_original": md.get("subject_original") or r.get("subject_original"),
        "difficulty": str(md.get("difficulty") or r.get("difficulty") or "unknown").lower(),
        "question": str(r["question"]).strip(),
        "choices": [str(c).strip() for c in r["choices"]],
        "answer": to_int(r.get("answer")),
    }

def base_problem(recs):
    """Detect numbering trouble that resolve_answer_index() only prints about."""
    vals = [r["answer"] for r in recs if r["answer"] is not None]
    max_n = max(len(r["choices"]) for r in recs)
    if 0 in vals and max_n in vals:
        return "INCONSISTENT: answers include both 0 and %d (mixed 0-based and 1-based?)" % max_n
    if 0 not in vals and max_n not in vals:
        return "AMBIGUOUS: no answer equals 0 or %d, so the base can't be inferred" % max_n
    return None

def diagnose_base(recs):
    """Show where answer==0 and answer==len(choices) occur, to find mixed-numbering groups."""
    for key in ("difficulty", "category", "subject"):
        g = collections.defaultdict(lambda: [0, 0, 0])
        for r in recs:
            a, k = r["answer"], r[key]
            g[k][2] += 1
            g[k][0] += int(a == 0)
            g[k][1] += int(a == len(r["choices"]))
        print(f"\nby {key}: group -> [#answer==0, #answer==num_options, total]")
        for k, v in sorted(g.items(), key=lambda kv: -kv[1][2])[:15]:
            print(f"  {k}: {v}")

""" function to stripping off unwanted MOE towers"""
def strip_to_text(model):
    inner = model.model
    for name, _ in list(inner.named_children()):
        if name != "language_model":     # vision_tower, audio_tower, their embedding projections, ...
            print("removing", name)
            delattr(inner, name)
    return model

""" body function that to the forward pass through the model weights """
def body(model_name="google/gemma-4-E4B-it",
         dataset_name="naist-nlp/SinhalaMMLU",
         config=None,            # dataset config name, if it has several
         split=None,             # the DEV split name (run once with None to see the options)
         n_sample=300,           # 0 = use every question
         perm=False,             # True = average over all cyclic option rotations
         intro=False,            # True = prepend the subject line
         quant="4bit",           # "4bit" or "none"
         answer_base=None,       # None = auto-detect, or force 0 / 1
         seed=0,
         out_dir="results",
         check_only=False):      # True = inspect data and the prompt, skip the model

    # 1. Load and normalise
    rows = load_hf(dataset_name, config, split)
    recs = [normalize(r, f"{dataset_name}:{split}") for r in rows
            if "question" in r and "choices" in r]
    if not recs:
        raise ValueError(f"No usable rows. Columns seen: {list(rows[0].keys())}")

    # 2. Answer numbering (an off-by-one silently ruins every score)
    resolve_answer_index(recs, answer_base)
    problem = None if answer_base in (0, 1) else base_problem(recs)
    if problem:
        diagnose_base(recs)
        raise ValueError(problem + "\nRead the diagnosis above, then rerun with answer_base=0 or 1 "
                         "(or fix the data).")

    valid = [r for r in recs if r["gold"] is not None and 0 <= r["gold"] < len(r["choices"])]
    if len(valid) < len(recs):
        print(f"WARNING: dropping {len(recs) - len(valid)} questions with a missing/out-of-range answer.")
    recs = valid
    if not recs:
        raise ValueError("Nothing left after validation. Check answer_base.")
    print("options per question:", dict(collections.Counter(len(r["choices"]) for r in recs)))
    print("difficulty:          ", dict(collections.Counter(r["difficulty"] for r in recs)))

    if n_sample and n_sample < len(recs):
        random.Random(seed).shuffle(recs)
        recs = recs[:n_sample]
        print(f"Using a random sample of {len(recs)} questions (seed {seed}).")

    r0 = recs[0]
    print("\n--- example prompt ---\n" + build_user(r0["question"], r0["choices"], r0["subject_original"], intro))
    if check_only:
        return recs

    # 3. Score
    model, tok = load_gemma4_8B(model_name, "none")
    digit_ids = digit_ids_for(tok)
    results, ntoks, t0 = [], [], time.time()
    for qi, r in enumerate(recs):
        n = len(r["choices"])
        rots = n if perm else 1
        acc_lp, plain = np.zeros(n), None
        for s in range(rots):
            order = [(i + s) % n for i in range(n)]      # slot i shows original option order[i]
            user = build_user(r["question"], [r["choices"][j] for j in order],
                              r["subject_original"], intro)

            lp, nt = digit_logprobs(model, tok, user, n, digit_ids)

            if s == 0:                                    # rotation 0 = original order
                plain = lp
                ntoks.append(nt)
            for slot, orig in enumerate(order):           # map slot scores back to original options
                acc_lp[orig] += lp[slot]
        r["plain_lp"], r["perm_lp"] = plain, acc_lp / rots
        r["plain"] = int(np.argmax(plain))
        r["perm"] = int(np.argmax(r["perm_lp"]))
        results.append(r)
        if (qi + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {qi + 1}/{len(recs)}  {el / (qi + 1):.2f}s/q  "
                  f"eta {el / (qi + 1) * (len(recs) - qi - 1) / 60:.1f} min")

    # 4. Calibration: remove each slot's average bias (offsets fitted on DEV only) --
    prior = {}
    for n in {len(r["choices"]) for r in results}:
        prior[n] = np.mean([r["plain_lp"] for r in results if len(r["choices"]) == n], axis=0)
    for r in results:
        r["calib"] = int(np.argmax(r["plain_lp"] - prior[len(r["choices"])]))

    # 5. Reports
    print_report(results, "plain", "PLAIN (original order, 1 pass)")
    print_report(results, "calib", "CALIBRATED (slot prior removed; fitted on dev)")
    if perm:
        print_report(results, "perm", "PERMUTATION-AVERAGED (all rotations)")
    print(f"\nprompt tokens: mean {np.mean(ntoks):.0f}, max {max(ntoks)}   "
          f"time {(time.time() - t0) / 60:.1f} min")

    # 6. Save per-question log-probs + one summary row per run
    os.makedirs(out_dir, exist_ok=True)
    tag = (model_name.replace("/", "_") + ("_perm" if perm else "") + ("_intro" if intro else ""))
    with open(os.path.join(out_dir, tag + ".jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            row = {k: r[k] for k in ("src", "q_no", "subject", "category", "difficulty",
                                     "gold", "plain", "calib", "perm")}
            row["plain_lp"], row["perm_lp"] = r["plain_lp"].tolist(), r["perm_lp"].tolist()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summ = os.path.join(out_dir, "summary.csv")
    new = not os.path.exists(summ)
    mean = lambda k: float(np.mean([r[k] == r["gold"] for r in results]))
    with open(summ, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "n", "plain", "calib", "perm", "quant", "intro", "mean_tokens"])
        w.writerow([tag, len(results), f"{mean('plain'):.4f}", f"{mean('calib'):.4f}",
                    f"{mean('perm'):.4f}" if perm else "", quant, intro, f"{np.mean(ntoks):.0f}"])
    print(f"Saved {tag}.jsonl and a row in {summ}")
    print("parameter count ",count_params(model_name))

    return model, tok, recs, digit_ids

model,tok, recs, digit_ids = body(model_name="google/gemma-4-E4B-it",split="train")