import torch
import re
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
import torch.distributed as dist
# import sleep
from time import sleep

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def extract_tag(text, left_tag, right_tag):
    """
    提取文本中位于 left_tag 和 right_tag 之间的内容。
    若任一标签缺失则返回空字符串。
    """
    left_pos = text.find(left_tag)
    if left_pos == -1:
        return ""
    right_pos = text.find(right_tag, left_pos + len(left_tag))
    if right_pos == -1:
        return ""
    return text[left_pos + len(left_tag): right_pos].strip()

def reward_func(queries, prompts, labels):
    """
    计算各个奖励分数：
      - 准确性奖励：比对 <answer> 中的答案与标签是否匹配；
      - 格式奖励：在整体格式满足预期的前提下给予奖励；
      - tag 数量奖励：检查答案中各标签数量是否正确。
    """
    responses = []
    for query, prompt in zip(queries, prompts):
        query = strip_sequence(query, "<|eot_id|>", "[PAD]")
        prompt = strip_sequence(prompt, "<|eot_id|>", "[PAD]")
        response = query[len(prompt):].strip()
        responses.append(response)
        
    labels = [eval(label) for label in labels]
    
    r_format = format_reward(responses)
    r_tag_count = tag_count_reward(responses)
    
    reactants = [label["reactants"] for label in labels]
    
    r_acc = accuracy_reward(responses, reactants)
    r_acc = torch.tensor(r_acc)
    r_format = (torch.tensor(r_format) + torch.tensor(r_tag_count)) / 2
    # 综合奖励：准确性奖励 + 格式奖励和 tag 数量奖励的平均
    rewards = r_acc + 0.5 * r_format

    return {
        "rewards": rewards,  # Rewards for advantage calculation
        "scores": torch.tensor(r_acc),  # Scores for dynamic filtering (0-1 reward)
        "extra_logs": {"format_reward": r_format},  # Additional logging info for wandb
    }

def accuracy_reward(responses, reactant_labels):
    """奖励函数：检查回答中中心和反应物是否与标签一致."""
    rewards = []
    for response, reactant_label in zip(responses, reactant_labels):
        reward = 0.0
        reactant_parsed = extract_tag(response, "<answer>", "</answer>")
        reward += float(verify(reactant_parsed, reactant_label))
        rewards.append(reward)
    return rewards

def format_reward(responses, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, response, re.DOTALL | re.MULTILINE) for response in responses]

    return [1.0 if match else 0.0 for match in matches]

def tag_count_reward(responses, **kwargs) -> list[float]:
    """
    tag 数量奖励函数：检查回答中各个标签出现的次数。
    每对标签计 0.25, 要求恰好出现 1 次。
    """
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count
    return [count_tags(res) for res in responses]

def verify(reactants, gt_reactants):
    """
    验证反应物 SMILES 字符串是否一致。首先调用 clear_map_canonical_smiles 对 SMILES 字符串做归一化处理，
    然后比对归一化结果是否一致。
    """
    try:
        pred_reactants_smi = clear_map_canonical_smiles(reactants, canonical=True)
        gt_reactants_smi = clear_map_canonical_smiles(gt_reactants, canonical=True)
        if pred_reactants_smi == "" or gt_reactants_smi == "":
            return 0.0
        else:
            return pred_reactants_smi == gt_reactants_smi
    except:
        return 0.0

_smiles_cache = {}
def clear_map_canonical_smiles(smi: str, canonical: bool = True) -> str:
    """将 SMILES 字符串转换为归一化的标准 SMILES 表示，过程中去除原子映射信息."""
    if smi in _smiles_cache:
        return _smiles_cache[smi]
    if len(smi) > 200:
        return ""
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        for atom in mol.GetAtoms():
            if atom.HasProp('molAtomMapNumber'):
                atom.ClearProp('molAtomMapNumber')
        result = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=canonical, kekuleSmiles=False)
        _smiles_cache[smi] = result
        return result
    return ""
