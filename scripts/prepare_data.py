import json
import argparse


def prepare_data(input_file, output_file):
    """Basic data preprocessing for instruction-following format"""
    processed = []
    
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'instruction' in data and 'output' in data:
                processed.append(data)
    
    with open(output_file, 'w') as f:
        for item in processed:
            f.write(json.dumps(item) + '\n')
    
    print(f"Processed {len(processed)} examples to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample.jsonl")
    parser.add_argument("--output", default="data/processed.jsonl")
    args = parser.parse_args()
    
    prepare_data(args.input, args.output)
