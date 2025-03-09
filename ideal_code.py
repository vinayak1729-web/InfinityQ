import os
import time
import re
import subprocess
import tempfile
from collections import Counter
import random
import pandas as pd
import polars as pl
import torch
from vllm import LLM, SamplingParams
import kaggle_evaluation.aimo_2_inference_server

# Import Dataset Prompt Functions
from prompt_2024 import all_prompts
from prompt_2023 import algebra_prompt as algebra_2023, geometry_prompt as geometry_2023, number_theory_prompt as number_theory_2023, combinatorics_prompt as combinatorics_2023
from prompt_2022 import algebra_prompt as algebra_2022, geometry_prompt as geometry_2022, number_theory_prompt as number_theory_2022, combinatorics_prompt as combinatorics_2022
from prompt_2021 import algebra_prompt as algebra_2021, geometry_prompt as geometry_2021, number_theory_prompt as number_theory_2021, combinatorics_prompt as combinatorics_2021
from prompt_2019 import algebra_prompt as algebra_2019, geometry_prompt as geometry_2019, number_theory_prompt as number_theory_2019, combinatorics_prompt as combinatorics_2019

# Environment Setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
start_time = time.time()
cutoff_time = start_time + 4.5 * 3600
cutoff_times = [start_time + i * (cutoff_time - start_time) / 50 for i in range(51)]

# Verify CUDA
print("CUDA Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())

# Model Initialization
MAX_NUM_SEQS = 16
model_path = "/kaggle/input/internlm2-math-plus-7b/internlm2-math-plus-7b"
llm = LLM(
    model=model_path,
    max_num_seqs=MAX_NUM_SEQS,
    max_model_len=8192,
    tensor_parallel_size=2,
    seed=2024,
    device="cuda"
)
tokenizer = llm.get_tokenizer()

# Dataset for Testing (2024)
TESTING_DATA = all_prompts()

# Python REPL
class PythonREPL:
    def __init__(self, timeout=5):
        self.timeout = timeout
    def __call__(self, code):
        code = "import math\n" + code.strip()
        if "print(" not in code.split("\n")[-1]:
            code += "\nprint(result)"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name
        try:
            result = subprocess.run(["python3", temp_file], capture_output=True, text=True, timeout=self.timeout)
            os.unlink(temp_file)
            if result.returncode == 0:
                return True, result.stdout.strip()
            return False, result.stderr.strip()
        except subprocess.TimeoutExpired:
            os.unlink(temp_file)
            return False, "Timeout"

def execute_code(text):
    executor = PythonREPL()
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if not code_blocks:
        return text, False
    success, output = executor(code_blocks[-1])
    return output, success

# Utilities
def extract_boxed_text(text):
    matches = re.findall(r'\\boxed{(.*?)}', text)
    return matches[-1] if matches else ""

def select_answer(answers):
    valid_answers = [int(a) for a in answers if a.strip().isdigit()]
    if not valid_answers:
        return 42
    counter = Counter(valid_answers)
    return counter.most_common(1)[0][0] % 1000

# Main Prompt: Analyze Question Type
def classify_question(question):
    question = question.lower()
    if "remainder" in question or "divided by" in question or "units digit" in question or "divisors" in question:
        return "number_theory"
    elif "triangle" in question or "area" in question or "circle" in question:
        return "geometry"
    elif "solve for" in question or "equation" in question or "roots" in question:
        return "algebra"
    elif "arrange" in question or "ways" in question:
        return "combinatorics"
    return "default"

# Hybrid Prompt with If-Else
def create_starter_messages(question, index, intuitive=False):
    q_type = classify_question(question)
    year = random.choice([2023, 2022, 2021, 2019])  # Randomly pick a year
    
    # Call type-specific prompt functions
    if q_type == "number_theory":
        training_example = {
            2023: number_theory_2023(),
            2022: number_theory_2022(),
            2021: number_theory_2021(),
            2019: number_theory_2019()
        }[year]
    elif q_type == "geometry":
        training_example = {
            2023: geometry_2023(),
            2022: geometry_2022(),
            2021: geometry_2021(),
            2019: geometry_2019()
        }[year]
    elif q_type == "algebra":
        training_example = {
            2023: algebra_2023(),
            2022: algebra_2022(),
            2021: algebra_2021(),
            2019: algebra_2019()
        }[year]
    elif q_type == "combinatorics":
        training_example = {
            2023: combinatorics_2023(),
            2022: combinatorics_2022(),
            2021: combinatorics_2021(),
            2019: combinatorics_2019()
        }[year]
    else:
        training_example = random.choice(TESTING_DATA)  # Fallback to 2024 data
    
    if intuitive:  # Ramanujan-style
        if q_type == "number_theory":
            prompt = f"Guess the answer using number patterns or cycles, like Ramanujan. Example: '{training_example}'. Return in \\boxed{{}}."
        elif q_type == "geometry":
            prompt = f"Estimate with symmetry or shape intuition, like Ramanujan. Example: '{training_example}'. Return in \\boxed{{}}."
        elif q_type == "algebra":
            prompt = f"Guess using equation patterns, like Ramanujan. Example: '{training_example}'. Return in \\boxed{{}}."
        elif q_type == "combinatorics":
            prompt = f"Guess using arrangement tricks, like Ramanujan. Example: '{training_example}'. Return in \\boxed{{}}."
        else:
            prompt = f"Guess creatively using patterns, like Ramanujan. Example: '{training_example}'. Return in \\boxed{{}}."
    else:  # Systematic style
        prompt = f"Solve step-by-step with Python code in ```python``` blocks. Verify rigorously. Example: '{training_example}'. Return in \\boxed{{}}."
    
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question}
    ]

def batch_generate(messages):
    max_tokens = 500 if time.time() < cutoff_times[-1] else 300
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        stop=["</think>", "```output"],
        include_stop_str_in_output=True
    )
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    outputs = llm.generate(texts, sampling_params)
    return [m + [{"role": "assistant", "content": o.outputs[0].text}] for m, o in zip(messages, outputs)]

# Prediction Logic
def predict_for_question(question: str) -> int:
    if time.time() > cutoff_time:
        return 42
    num_seqs = MAX_NUM_SEQS if time.time() < cutoff_times[-1] else 12
    half_seqs = num_seqs // 2
    
    messages = ([create_starter_messages(question, i, intuitive=True) for i in range(half_seqs)] + 
                [create_starter_messages(question, i, intuitive=False) for i in range(half_seqs, num_seqs)])
    answers = []
    
    for _ in range(2):
        messages = batch_generate(messages)
        for i, msg in enumerate(messages):
            response = msg[-1]["content"]
            if not intuitive or "```python" in response:
                output, success = execute_code(response)
                if success:
                    boxed = extract_boxed_text(output) or output.strip()
                    if boxed:
                        answers.append(boxed)
                messages[i] = msg + [{"role": "assistant", "content": output if success else "Retry"}]
            else:
                boxed = extract_boxed_text(response)
                if boxed:
                    answers.append(boxed)
                messages[i] = msg
        if len(answers) >= half_seqs:
            break
    
    print(f"Question: {question}, Answers: {answers}")
    return select_answer(answers)

def predict(id_: pl.DataFrame, question: pl.DataFrame) -> pl.DataFrame:
    id_ = id_.item(0)
    question = question.item(0)
    print(f"Processing ID: {id_}")
    answer = predict_for_question(question)
    print(f"Predicted Answer: {answer}")
    return pl.DataFrame({'id': id_, 'answer': answer})

# Run
inference_server = kaggle_evaluation.aimo_2_inference_server.AIMO2InferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    ref_df = pd.read_csv('/kaggle/input/ai-mathematical-olympiad-progress-prize-2/reference.csv')
    ref_df[['id', 'question']].to_csv('reference.csv', index=False)
    inference_server.run_local_gateway('reference.csv')