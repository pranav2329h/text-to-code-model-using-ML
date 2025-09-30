import os, re, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM = (
    "You output ONLY a single complete HTML document.\n"
    "- Must include <!DOCTYPE html>, <style> (inline CSS), and <script> (inline JS).\n"
    "- No markdown fences, no explanations, HTML only."
)

USER_PROMPT = "Landing page with title 'My App', a paragraph, and a centered email signup form."

def main():
    os.makedirs("preview", exist_ok=True)
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cpu")

    # Use the model's chat template for best results
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tok(prompt, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=700,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    text = tok.decode(out[0], skip_special_tokens=True)

    # Save RAW text so we can inspect what the model actually produced
    with open("preview/raw.txt", "w", encoding="utf-8") as f:
        f.write(text)

    # Try to extract a full HTML doc
    m = re.search(r'<!DOCTYPE\s+html[\s\S]*?</html>', text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r'<html[\s\S]*?</html>', text, flags=re.IGNORECASE)
    html = (m.group(0) if m else text).strip()

    # Fallback if still empty: wrap whatever we got into a minimal HTML shell
    if not html.strip():
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Fallback</title>
<style>body{{font-family:sans-serif;padding:24px}}</style></head>
<body><h1>Fallback</h1><p>The model returned no HTML tags.</p>
<pre>{text}</pre>
<script></script></body></html>"""

    with open("preview/output.html", "w", encoding="utf-8") as f:
        f.write(html)

    print(f"RAW length: {len(text)} chars")
    print(f"HTML length: {len(html)} chars")
    print("Wrote preview/raw.txt and preview/output.html")

if __name__ == "__main__":
    main()
