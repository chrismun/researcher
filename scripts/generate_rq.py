from transformers import pipeline, set_seed
from search_arxiv import *

def generate_research_questions(summary, model_name='EleutherAI/gpt-neo-2.7B', num_questions=5):
    generator = pipeline('text-generation', model=model_name)
    set_seed(42)
    prompt = f"Based on the following summary, generate {num_questions} research questions:\n\n{summary}\n\nQuestions:"
    questions = generator(prompt, max_length=200, num_return_sequences=1)
    print(questions[0]['generated_text'])

paper = search_arxiv("OpenACC Validation and Verification")
generate_research_questions(paper )
