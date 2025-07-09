import json 
import argparse
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)

def prepare_post_beam_search(tokenizer_path, input_file, output_file):
    """
    Prepare post beam search data
    
    Args:
        tokenizer_path (str): tokenizer model path
        input_file (str): input file path
        output_file (str): output file path
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Read data
    data = []
    with open(input_file, "r") as f:
        for line in f:
            data.append(json.loads(line))

    prod_react = {}
    post_data = []
    
    # Process data
    for item in data:
        prod = item["input"].split("<SMILES>")[1].split("</SMILES>")[0].strip()
        if prod not in prod_react:
            prod_react[prod] = []
        for pred in item["pred"]: 
            chat = [{"role": "user", "content": item["input"]},{"role": "assistant", "content": pred}]
            chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            prompt = chat.rsplit("<answer>",1)[0] + "<answer>\n"

            post_data.append({
                "input": prompt,
                "gold": item["gold"]
            })

    # Write to output file
    with open(output_file, "w") as f:
        for item in post_data:
            f.write(json.dumps(item) + "\n")
    
    print(f"Processing completed! Output file: {output_file}")
    print(f"Total processed {len(post_data)} records")

def main():
    parser = argparse.ArgumentParser(description='Prepare post beam search data')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='tokenizer model path')
    parser.add_argument('--input_file', type=str, required=True, help='input file path')
    parser.add_argument('--output_file', type=str, required=True, help='output file path')
    
    args = parser.parse_args()
    
    prepare_post_beam_search(args.tokenizer_path, args.input_file, args.output_file)

if __name__ == "__main__":
    main()
