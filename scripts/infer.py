import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    model = PeftModel.from_pretrained(base, args.lora)

    prompt = args.prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()

    main(args)
