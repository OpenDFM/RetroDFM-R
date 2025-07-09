import argparse
import os
import json
from tqdm import tqdm
import torch.multiprocessing as mp
from queue import Empty
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=1,
        help='Specify sample number',
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help='Specify temperature',
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help='Specify max new tokens',
    )
    parser.add_argument(
        '--input_file',
        type=str,
        help="input file path",
        required=True,
    )
    parser.add_argument(
        '--output_file',
        type=str,
        help='output file path',
        required=True,
    )
    parser.add_argument(
        '--use_beam_search',
        action='store_true'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='0',
        help='GPU IDs separated by comma'
    )
    parser.add_argument(
        '--post_beam',
        action='store_true'
    )
    args = parser.parse_args()
    return args

def prompt_input(text, tokenizer, post_beam=False):
    if post_beam:
        return text
    else:
        message = [{"role": "user", "content": text}]
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

def evaluate_task(gpu_id, task_queue, result_queue, model_path, args):
    """Evaluate tasks on specified GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running on GPU {gpu_id}")
    
    model = LLM(model=model_path, enable_prefix_caching=True, max_model_len=2048)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def generate(input_texts):
        sampling_params = SamplingParams(
            n=args.sample_num, 
            best_of=args.sample_num,
            temperature=args.temperature,
            skip_special_tokens=True,
            max_tokens=args.max_new_tokens,
            top_k=-1, 
            top_p=1,
            stop='<|im_end|>',
            stop_token_ids=[tokenizer.eos_token_id],
            use_beam_search=args.use_beam_search
        )
        outputs = model.generate(input_texts, sampling_params, use_tqdm=True)
        return outputs

    while True:
        try:
            batch_idx, batch_prompts = task_queue.get(timeout=5)  # 5 second timeout
            if batch_idx == -1:  # End signal
                break
            results = generate(batch_prompts)
            result_queue.put((batch_idx, results))
        except Empty:
            break
        except Exception as e:
            print(f"GPU {gpu_id} error: {e}")
            result_queue.put((batch_idx, None))

def main():
    args = parse_args()
    print(f"args: {args}")

    # Read processed SMILES (if output_file exists)
    processed_smiles = set()
    if os.path.exists(args.output_file):
        print(f"Found existing output file, reading processed samples...")
        with open(args.output_file, "r") as reader:
            for line in reader:
                try:
                    result_data = json.loads(line.strip())
                    if 'input' in result_data:
                        input_text = result_data['input']
                        # Extract SMILES
                        if "<SMILES>" in input_text and "</SMILES>" in input_text:
                            smiles = input_text.split("<SMILES>")[1].split("</SMILES>")[0].strip()
                            processed_smiles.add(smiles)
                except:
                    continue
        print(f"Found {len(processed_smiles)} processed samples")

    data = []
    with open(args.input_file, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    input_texts = []
    examples = []
    skipped_count = 0
    
    for idx, datum in enumerate(tqdm(data, desc="Filtering unprocessed samples")):
        # Check if already processed
        input_text = datum['input']
        should_skip = False
        
        if "<SMILES>" in input_text and "</SMILES>" in input_text:
            smiles = input_text.split("<SMILES>")[1].split("</SMILES>")[0].strip()
            if smiles in processed_smiles:
                should_skip = True
                skipped_count += 1
        
        if not should_skip:
            input_texts.append(prompt_input(datum['input'], tokenizer, args.post_beam))
            example = {}
            example['input'] = datum['input']
            if 'gold' not in datum:
                example['gold'] = datum['label']["reactants"]
                if "center" in datum['label']:
                    example['center'] = datum['label']['center']
            else:
                example['gold'] = datum['gold']
            examples.append(example)
    
    print(f"Skipped {skipped_count} processed samples")
    print(f"Need to process {len(input_texts)} new samples")
    # If no samples need processing, exit directly
    if len(input_texts) == 0:
        print("All samples have been processed!")
        return

    # Setup multiprocessing
    gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
    mp.set_start_method('spawn', force=True)
    
    # Create task queue and result queue
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    # Dynamic allocation of mini-batch tasks
    # MINI_BATCH_SIZE = len(input_texts) // len(gpu_ids) + 1 # Dynamically adjust batch size based on GPU count
    MINI_BATCH_SIZE = 1024
    num_batches = (len(input_texts) + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE
    
    for i in range(num_batches):
        start_idx = i * MINI_BATCH_SIZE
        end_idx = min(start_idx + MINI_BATCH_SIZE, len(input_texts))
        task_queue.put((i, input_texts[start_idx:end_idx]))
    
    # Add end signals
    for _ in range(len(gpu_ids)):
        task_queue.put((-1, None))
    
    # Start GPU processes
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=evaluate_task,
                      args=(gpu_id, task_queue, result_queue, args.model_name_or_path, args))
        processes.append(p)
        p.start()

    # Create output directory (if not exists)
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))
        
    # If output_file doesn't exist, create empty file; if exists, keep original content
    if not os.path.exists(args.output_file):
        with open(args.output_file, "w") as writer:
            pass  # Create empty file
    
    # Create index mapping to track batch results
    batch_index_map = {}
    for i in range(num_batches):
        start_idx = i * MINI_BATCH_SIZE
        end_idx = min(start_idx + MINI_BATCH_SIZE, len(input_texts))
        batch_index_map[i] = (start_idx, end_idx)
    
    # Collect results
    completed_batches = 0
    total_examples = len(input_texts)
    
    # Create two progress bars
    batch_pbar = tqdm(total=num_batches, desc="Batch processing progress")
    total_pbar = tqdm(total=total_examples, desc="Overall processing progress")
    
    # Create set of processed batches to ensure sequential writing
    processed_batches = set()
    next_batch_to_write = 0
    pending_results = {}
    
    while completed_batches < num_batches:
        batch_idx, batch_results = result_queue.get()
        completed_batches += 1
        batch_pbar.update(1)
        
        if batch_results is not None:
            # Update processed sample count
            total_pbar.update(len(batch_results))
            
            # Save results for sequential writing
            pending_results[batch_idx] = batch_results
            processed_batches.add(batch_idx)
            
            # Try to write files sequentially
            while next_batch_to_write in processed_batches:
                # Get sample index range for current batch
                start_idx, end_idx = batch_index_map[next_batch_to_write]
                current_results = pending_results[next_batch_to_write]
                
                # Process and write results
                with open(args.output_file, "a") as writer:
                    for i, result in enumerate(current_results):
                        example_idx = start_idx + i
                        if example_idx < len(examples):
                            examples[example_idx]["pred"] = []
                            for j in range(len(result.outputs)):
                                text = result.outputs[j].text.split("<|eot_id|>")[0].split("<|im_end|>")[0].strip()
                                examples[example_idx]["pred"].append(text)
                            # Write result for this sample
                            writer.write(json.dumps(examples[example_idx]))
                            writer.write("\n")
                
                # Clean up memory
                del pending_results[next_batch_to_write]
                next_batch_to_write += 1
    
    batch_pbar.close()
    total_pbar.close()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    print(f"All results have been incrementally written to file: {args.output_file}")

if __name__ == "__main__":
    main()
