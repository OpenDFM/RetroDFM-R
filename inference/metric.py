import re
import json
import random
import numpy as np
from tqdm import tqdm
from rdkit import Chem 
import multiprocessing as mp
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Set random seed
random.seed(42)
np.random.seed(42)

Alpha = 1

class Skip:
    @staticmethod
    def decode(x):
        return x


def smiles_em(gold, pred):
    return gold == pred


def smiles_validity(converter, pred):
    for p in pred:
        try:
            assert Chem.MolFromSmiles(converter.decode(p)) is not None
        except Exception:
            return 0
    return 1

def get_gold_pred(gold,pred):
    try:
        if "<answer>" in pred:
            pred = pred.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            pred = pred.split("</answer>")[0].strip()
    except:
        pred = ""
    try:
        gold = Chem.MolToSmiles(Chem.MolFromSmiles(gold), isomericSmiles=True)
    except:
        gold = ""
    try:
        pred = Chem.MolToSmiles(Chem.MolFromSmiles(pred), isomericSmiles=True)
    except:
        pred = ""
    return gold, pred

def analyze_molecular_properties(prod_smiles, gold_smiles):
    """Analyze molecular chirality and ring structure properties"""
    try:
        prod_mol = Chem.MolFromSmiles(prod_smiles)
        gold_mol = Chem.MolFromSmiles(gold_smiles)
        
        if prod_mol is None or gold_mol is None:
            return False, False, False, 0
        
        # Check chirality
        has_chirality = prod_smiles.count("@") > 0 or gold_smiles.count("@") > 0
        
        # Check ring structures
        prod_ringinfo = prod_mol.GetRingInfo()
        gold_ringinfo = gold_mol.GetRingInfo()
        prod_ringnum = prod_ringinfo.NumRings()
        gold_ringnum = gold_ringinfo.NumRings()
        
        # Determine ring opening and ring formation
        ring_opening = prod_ringnum < gold_ringnum  # Fewer rings in product than reactant indicates ring opening
        ring_formation = prod_ringnum > gold_ringnum  # More rings in product than reactant indicates ring formation
        
        # Calculate atom count difference
        atom_size_diff = len(gold_mol.GetAtoms()) - len(prod_mol.GetAtoms())
        
        return has_chirality, ring_opening, ring_formation, atom_size_diff
    except:
        return False, False, False, 0

def process_sample(sample_line):
    """Process a single sample function for multiprocessing"""
    try:
        sample = json.loads(sample_line.strip())
        
        # Check data format
        if "input" not in sample or "gold" not in sample or "pred" not in sample:
            return None
            
        prod = sample["input"].split('<SMILES>')[1].split('</SMILES>')[0]
        prod = Chem.MolToSmiles(Chem.MolFromSmiles(prod))
        gold = sample['gold']
        try:
            gold_center = sample['center']
        except:
            gold_center = ""
        
        # Analyze molecular properties
        has_chirality, ring_opening, ring_formation, atom_size_diff = analyze_molecular_properties(prod, gold)
        
        preds = {}
        centers = {}
        for i, pred_text in enumerate(sample['pred']):
            gold_i, pred_i = get_gold_pred(gold, pred_text.strip())
            preds[pred_i] = preds.get(pred_i, 0) + 1 / (1 + Alpha * i)
            if pred_i == "":
                continue
            try:
                center = pred_text.split('<center>')[1].split('</center>')[0].strip()
            except:
                center = ""
            centers[center] = centers.get(center, 0) + 1
        preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
        centers = sorted(centers.items(), key=lambda x: x[1], reverse=True)
        counted_preds = []
        for i , (pred, count) in enumerate(preds):
            counted_preds.append((pred, count))
        
        return prod, gold_i, counted_preds, gold_center, centers, has_chirality, ring_opening, ring_formation, atom_size_diff
    except Exception as e:
        print(f"Error processing sample: {e}")
        return None

def evaluate(outputs, detailed=True, rxn_type_dict=None):
    print("Starting evaluation...")
    total = 0
    metrics = {}
    
    # Basic metrics
    for i in [1, 3, 5, 10]:
        metrics[f'coverage_{i}'] = 0
        metrics[f'majority_{i}'] = 0
        metrics[f'center_{i}'] = 0
        
        # Detailed metrics - count correct predictions for each category separately
        if detailed:
            metrics[f'chirality_correct_{i}'] = 0
            metrics[f'no_chirality_correct_{i}'] = 0
            metrics[f'ring_opening_correct_{i}'] = 0
            metrics[f'ring_formation_correct_{i}'] = 0
            metrics[f'no_ring_change_correct_{i}'] = 0
    
    # Reaction type metrics initialization
    rxn_type_metrics = {}
    rxn_type_counts = {}
    if rxn_type_dict:
        unique_types = set(rxn_type_dict.values())
        for rxn_type in unique_types:
            rxn_type_counts[rxn_type] = 0
            for i in [1, 3, 5, 10]:
                rxn_type_metrics[f'{rxn_type}_correct_{i}'] = 0
    
    # Statistical counters
    total_chirality = 0
    total_ring_opening = 0
    total_ring_formation = 0
    atom_size_diffs = []
    
    prods_result = {}

    print("Processing samples...")
    # Use multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_sample, outputs), total=len(outputs), desc="Processing progress"))
    
    # Process results - ensure each product is processed only once
    for result in results:
        if result is None:
            continue
            
        prod, gold, preds, gold_center, centers, has_chirality, ring_opening, ring_formation, atom_size_diff = result
        if prod not in prods_result:
            prods_result[prod] = {
                'gold': gold,
                'gold_center': gold_center,
                'pred': {},
                'centers': centers,
                'has_chirality': has_chirality,
                'ring_opening': ring_opening,
                'ring_formation': ring_formation,
                'atom_size_diff': atom_size_diff
            }
        # Merge prediction results
        for pred in preds:
            prods_result[prod]['pred'][pred[0]] = prods_result[prod]['pred'].get(pred[0], 0) + pred[1]
    
    print("Calculating metrics...")
    for prod, result in tqdm(prods_result.items(), desc="Calculating metrics"):
        gold = result['gold']
        gold_center = result['gold_center']
        preds = result['pred'].items()
        preds = sorted(preds, key=lambda x: x[1], reverse=True)
        centers = result['centers']
        
        # Molecular properties
        has_chirality = result['has_chirality']
        ring_opening = result['ring_opening']
        ring_formation = result['ring_formation']
        atom_size_diff = result['atom_size_diff']
        
        # Get reaction type
        rxn_type = None
        if rxn_type_dict and prod in rxn_type_dict:
            rxn_type = rxn_type_dict[prod]
            rxn_type_counts[rxn_type] += 1
        
        total += 1
        
        # Count molecular properties
        if has_chirality:
            total_chirality += 1
        if ring_opening:
            total_ring_opening += 1
        if ring_formation:
            total_ring_formation += 1
        atom_size_diffs.append(atom_size_diff)
        
        # Find the position of correct answer in predictions
        correct_rank = -1
        
        # gold = sorted(gold.split('.'),reverse=True,key=lambda x: len(x))[0]

        for k, (pred, count) in enumerate(preds):
            # pred = sorted(pred.split('.'),reverse=True,key=lambda x: len(x))[0]
            if pred == gold:
                correct_rank = k
                break
        
        # Calculate various top-k metrics
        for j in [1, 3, 5, 10]:
            # Coverage: whether correct answer is in top-j
            is_correct_in_topj = correct_rank != -1 and correct_rank < j
            if is_correct_in_topj:
                metrics[f'coverage_{j}'] += 1
                
                # Reaction type metrics
                if rxn_type:
                    rxn_type_metrics[f'{rxn_type}_correct_{j}'] += 1
                
                # Detailed metrics: only calculate when prediction is correct, each sample counted once
                if detailed:
                    if has_chirality:
                        metrics[f'chirality_correct_{j}'] += 1
                    else:
                        metrics[f'no_chirality_correct_{j}'] += 1
                        
                    if ring_opening:
                        metrics[f'ring_opening_correct_{j}'] += 1
                    elif ring_formation:
                        metrics[f'ring_formation_correct_{j}'] += 1
                    else:
                        metrics[f'no_ring_change_correct_{j}'] += 1
            
            # Majority: whether the highest frequency prediction is correct
            if preds and preds[0][0] == gold:
                metrics[f'majority_{j}'] += 1
            
            # Center: whether the highest frequency center prediction is correct
            if centers and centers[0][0] == gold_center:
                metrics[f'center_{j}'] += 1

    # Avoid division by zero
    if total == 0:
        print("Warning: No samples processed successfully!")
        return metrics
    
    # Calculate percentages - collect all keys first to avoid modifying dict during iteration
    keys_to_process = list(metrics.keys())
    
    # Process basic metrics first
    for key in keys_to_process:
        if key.startswith(('coverage_', 'majority_', 'center_')) and not key.endswith('correct_'):
            metrics[key] = metrics[key] / total * 100
    
    # Process detailed metrics
    if detailed:
        for k in [1, 3, 5, 10]:
            # Chirality accuracy
            if f'chirality_correct_{k}' in metrics:
                metrics[f'chirality_{k}'] = metrics[f'chirality_correct_{k}'] / max(total_chirality, 1) * 100
            
            # Non-chirality accuracy
            if f'no_chirality_correct_{k}' in metrics:
                metrics[f'no_chirality_{k}'] = metrics[f'no_chirality_correct_{k}'] / max(total - total_chirality, 1) * 100
            
            # Ring opening accuracy
            if f'ring_opening_correct_{k}' in metrics:
                metrics[f'ring_opening_{k}'] = metrics[f'ring_opening_correct_{k}'] / max(total_ring_opening, 1) * 100
            
            # Ring formation accuracy
            if f'ring_formation_correct_{k}' in metrics:
                metrics[f'ring_formation_{k}'] = metrics[f'ring_formation_correct_{k}'] / max(total_ring_formation, 1) * 100
            
            # No ring change accuracy
            if f'no_ring_change_correct_{k}' in metrics:
                no_ring_change_total = total - total_ring_opening - total_ring_formation
                metrics[f'no_ring_change_{k}'] = metrics[f'no_ring_change_correct_{k}'] / max(no_ring_change_total, 1) * 100
    
    # Process reaction type metrics
    if rxn_type_dict:
        for rxn_type in rxn_type_counts:
            type_count = rxn_type_counts[rxn_type]
            if type_count > 0:
                for k in [1, 3, 5, 10]:
                    correct_count = rxn_type_metrics.get(f'{rxn_type}_correct_{k}', 0)
                    metrics[f'{rxn_type}_{k}'] = correct_count / type_count * 100
    
    # Add statistical information
    metrics['total_samples'] = total
    metrics['total_chirality'] = total_chirality
    metrics['total_ring_opening'] = total_ring_opening
    metrics['total_ring_formation'] = total_ring_formation
    metrics['avg_atom_size_diff'] = np.mean(atom_size_diffs) if atom_size_diffs else 0
    
    # Add reaction type statistics
    if rxn_type_dict:
        metrics['rxn_type_counts'] = rxn_type_counts
    
    # Debug information
    if detailed:
        print("\nDebug information:")
        for k in [1]:
            print(f"Top-{k} chirality correct count: {metrics.get(f'chirality_correct_{k}', 0)}")
            print(f"Top-{k} non-chirality correct count: {metrics.get(f'no_chirality_correct_{k}', 0)}")
            print(f"Total chirality samples: {total_chirality}")
            print(f"Total non-chirality samples: {total - total_chirality}")
    
    print("Evaluation completed!")
    return metrics

if __name__ == '__main__':
    import json
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='Prediction result file path')
    parser.add_argument('--detailed', action='store_true', default=True, help='Whether to calculate detailed metrics')
    args = parser.parse_args()
    
    with open("rxn_type_dict.json", 'r') as f:
        rxn_type_dict = json.load(f)

    
    # Load prediction data
    outputs = []
    with open(args.pred) as f:
        for line in f:
            outputs.append(line)

    
    print(f"Loaded {len(outputs)} prediction samples")
    
    # Evaluate
    metrics = evaluate(outputs, detailed=args.detailed, rxn_type_dict=rxn_type_dict)
    
    # Print results
    print("\n=== Basic Metrics ===")
    for k in [1, 3, 5, 10]:
        print(f"Top-{k} Coverage: {metrics[f'coverage_{k}']:.2f}%")
        print(f"Top-{k} Majority: {metrics[f'majority_{k}']:.2f}%")
        print(f"Top-{k} Center: {metrics[f'center_{k}']:.2f}%")
        print()
    
    if args.detailed:
        print("=== Detailed Metrics ===")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Chirality samples: {metrics['total_chirality']}")
        print(f"Ring opening samples: {metrics['total_ring_opening']}")
        print(f"Ring formation samples: {metrics['total_ring_formation']}")
        print(f"Average atom count difference: {metrics['avg_atom_size_diff']:.2f}")
        print()
        
        for k in [1, 3, 5, 10]:
            print(f"Top-{k} Chirality accuracy: {metrics[f'chirality_{k}']:.2f}%")
            print(f"Top-{k} Non-chirality accuracy: {metrics[f'no_chirality_{k}']:.2f}%")
            print(f"Top-{k} Ring opening accuracy: {metrics[f'ring_opening_{k}']:.2f}%")
            print(f"Top-{k} Ring formation accuracy: {metrics[f'ring_formation_{k}']:.2f}%")
            print(f"Top-{k} No ring change accuracy: {metrics[f'no_ring_change_{k}']:.2f}%")
            print()
    
    # Print reaction type metrics
    if rxn_type_dict and 'rxn_type_counts' in metrics:
        print("=== Reaction Type Metrics ===")
        rxn_type_counts = metrics['rxn_type_counts']
        
        # Sort by sample count for display
        sorted_types = sorted(rxn_type_counts.items(), key=lambda x: x[1], reverse=True)
        
        for rxn_type, count in sorted_types:
            print(f"\n{rxn_type} (sample count: {count})")
            for k in [1, 3, 5, 10]:
                accuracy = metrics.get(f'{rxn_type}_{k}', 0)
                print(f"  Top-{k}: {accuracy:.2f}%")
    
    print("\nAll metrics:", metrics) 
