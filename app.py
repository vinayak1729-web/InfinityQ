import sys
import random

# Add Prompt Directories to Path
sys.path.append("prompts/2019")
sys.path.append("prompts/2021")
sys.path.append("prompts/2022")
sys.path.append("prompts/2023")
sys.path.append("prompts/2024")

# Import Prompt Functions
from algebra2019 import algebra_prompt as algebra_2019
from combinatorsics2019 import combinators_prompt as combinatorics_2019
from geometry2019 import geometry_prompt as geometry_2019
from NumberTheory2019 import NumberTheory as number_theory_2019

from algebra2021_trainquestions import algebra2021 as algebra_2021
from combinatrics2021_trainquestions import combinatrics2021 as combinatorics_2021
from geometry2021_trainquestions import geometry2021 as geometry_2021
from numbertheory2021_trainquestions import numbertheory as number_theory_2021

from algebra2022trainquestions import algebra2022 as algebra_2022
from combinatrics2022_trainquestions import combinatrics2022_trainquestions as combinatorics_2022
from geometry2022_trainquestions import geometry2022 as geometry_2022
from numbertheory2022_trainquestions import numbertheory2022 as number_theory_2022

from algebra_train_questions2023 import algebraPrompt as algebra_2023
from Combinatorics_train_questions2023 import Combinatorics as combinatorics_2023
from geometry_train_questions2023 import geometry as geometry_2023
from numbertheory_train_questions2023 import numbertheory as number_theory_2023

from all_questions import train_q_2024

# Testing Data (2024)
TESTING_DATA = train_q_2024()

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
    year = random.choice([2019, 2021, 2022, 2023])
    
    # Dynamic Prompt Selection
    if q_type == "algebra":
        training_example = {
            2019: algebra_2019(),
            2021: algebra_2021(),
            2022: algebra_2022(),
            2023: algebra_2023()
        }[year]
    elif q_type == "geometry":
        training_example = {
            2019: geometry_2019(),
            2021: geometry_2021(),
            2022: geometry_2022(),
            2023: geometry_2023()
        }[year]
    elif q_type == "number_theory":
        training_example = {
            2019: number_theory_2019(),
            2021: number_theory_2021(),
            2022: number_theory_2022(),
            2023: number_theory_2023()
        }[year]
    elif q_type == "combinatorics":
        training_example = {
            2019: combinatorics_2019(),
            2021: combinatorics_2021(),
            2022: combinatorics_2022(),
            2023: combinatorics_2023()
        }[year]
    else:
        training_example = random.choice(TESTING_DATA)
    
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

# Main Function to Print Prompt
def main():
    # Example question (you can change this)
    question = "Find the remainder when 2^100 is divided by 7"
    
    # Print both intuitive and systematic prompts
    intuitive_prompt = create_starter_messages(question, 0, intuitive=True)
    systematic_prompt = create_starter_messages(question, 0, intuitive=False)
    
    print("Intuitive Prompt (Ramanujan-style):")
    print(intuitive_prompt)
    print("\nSystematic Prompt (Professional-style):")
    print(systematic_prompt)
