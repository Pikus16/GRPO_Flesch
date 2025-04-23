import unsloth
from unsloth import FastLanguageModel
import torch
import textstat
from vllm import SamplingParams
import os
from trl import GRPOConfig, GRPOTrainer
import click
import re
from datasets import load_dataset, Dataset

# ---------- Constants ----------
SYSTEM_PROMPT = """
Respond in the following format. Make sure to include the reasoning and answer tags.
<reasoning>
REASONING
</reasoning>
<answer>
INTEGER ANSWER
</answer>
""".strip()

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# ---------- Utility Functions ----------
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1].split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    return text.split("####")[-1].strip() if "####" in text else None

def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125 - len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125 - (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

# # ---------- Reward Functions ----------
# def flesch_kincaid_reward_func(completions, **kwargs):
#     responses = [c[0]['content'] for c in completions]
#     responses = [c.split("<reasoning>")[-1].split("</reasoning>")[0] for c in responses]
#     scores = [textstat.flesch_kincaid_grade(r) for r in responses]
#     return [3 if s <= 6 else 2 if s < 8 else 1 if s < 10 else 0 for s in scores]

def flesch_kincaid_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    reasoning_pattern = r"<reasoning>\n(.*?)\n</reasoning>"
    final_scores = []

    for response in responses:
        match = re.search(reasoning_pattern, response, flags=re.DOTALL)
        if match:
            reasoning_text = match.group(1).strip()

            # Word count check
            if len(reasoning_text.split()) < 10:
                final_scores.append(0.0)  # Too short to be meaningful
                continue

            # Flesch-Kincaid Grade (lower is easier to read)
            fk_grade = textstat.flesch_kincaid_grade(reasoning_text)

            # Smooth inverse scaling: lower grade → higher reward (max at FK = 0, min at FK = 10+)
            normalized_score = max(0.0, min(10.0 - fk_grade, 10.0)) / 10.0
            reward = normalized_score * 2
            final_scores.append(reward)
        else:
            final_scores.append(0.0)  # No <reasoning> tag found
    return final_scores


def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [c[0]['content'] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

def int_reward_func(completions, **kwargs):
    extracted = [extract_xml_answer(c[0]['content']) for c in completions]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]

def strict_format_reward_func(completions, **kwargs):
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, flags=re.DOTALL) else 0.0 for r in responses]

def soft_format_reward_func(completions, **kwargs):
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, flags=re.DOTALL) else 0.0 for r in responses]

def xmlcount_reward_func(completions, **kwargs):
    contents = [c[0]["content"] for c in completions]
    return [count_xml(c) for c in contents]

# ---------- Main Functions ----------
def load_model_and_tokenizer(max_seq_length: int = 2048, lora_rank: int = 16):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.8,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    return model, tokenizer

def train(model, tokenizer, dataset,
          save_path: str,
          max_seq_length: int = 2048,
          max_prompt_length: int = 256):
    config = GRPOConfig(
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_generations=4,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_seq_length - max_prompt_length,
        num_train_epochs=1,
        max_grad_norm=0.1,
        report_to="wandb",
        output_dir="outputs",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
            flesch_kincaid_reward_func,
        ],
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    model.save_lora(f"{save_path}/lora")
    model.save_pretrained_merged(f"{save_path}/merged", tokenizer, save_method="lora")


def run_inference(model, tokenizer, save_path):
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Calculate pi."},
    ], tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
    output = model.fast_generate(
        prompt,
        sampling_params=sampling_params,
        lora_request=model.load_lora(f"{save_path}/lora"),
    )[0].outputs[0].text
    print("INFERENCE:\n", output)

def setup_wandb(project='GRPO_FLESCH', name='gsm8k'):
    os.environ['WANDB_PROJECT'] = project
    os.environ['WANDB_NAME'] = name

@click.command()
@click.option('--project', type=str, default='GRPO_Flesch')
@click.option('--name', type=str, default='gsm8k')
@click.option('--save_path', type=str, default='output')
def main(project, name, save_path):
    #os.environ['WANDB_API_KEY'] = '54a50cbe22da3f857fcb7812dd80fedc2ef01ad4'
    setup_wandb(project=project, name=name)
    
    model, tokenizer = load_model_and_tokenizer()
    dataset = get_gsm8k_questions()

    train(model, tokenizer, dataset, save_path=save_path)
    run_inference(model, tokenizer, save_path=save_path)

if __name__ == '__main__':
    main()

