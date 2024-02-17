from transformers import pipeline, set_seed
from xml.etree import ElementTree
import subprocess
import requests

TOPIC = "OpenACC Validation and Verification"

def search_arxiv(keywords):
    query = '+AND+'.join(keywords.split())
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1'
    response = requests.get(url)
    root = ElementTree.fromstring(response.content)

    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        print(f"Title: {title}\nSummary:\n{summary}\n")

    return f"Title: {title}\nSummary:\n{summary}\n"


def generate_research_question(summary, model_name='EleutherAI/gpt-neo-2.7B', num_questions=1):
    generator = pipeline('text-generation', model=model_name)
    set_seed(42)
    prompt = f"Based on the following paper, generate {num_questions} research question:\n\n{summary}\n\nQuestions:"
    questions = generator(prompt, max_length=200, num_return_sequences=1)
    print(questions[0]['generated_text'])
    return questions[0]['generated_text']

def generate_research_plan(research_question, model_name='EleutherAI/gpt-neo-2.7B'):
    generator = pipeline('text-generation', model=model_name)
    set_seed(42)
    prompt = f"""
    # Given the research question: "{research_question}", generate a detailed research plan. The plan should outline the objectives, methodologies, tools (e.g., PyTorch, NAMD), datasets, and evaluation metrics to be used. Consider that the execution will be performed on a typical compute cluster with command-line capabilities.
    
    # Research Question: {research_question}
    """
    research_plan = generator(prompt, max_length=1000, num_return_sequences=1)[0]['generated_text']
    print(research_plan)
    return research_plan


def generate_experiment(research_plan, model_name='EleutherAI/gpt-neo-2.7B'):
    generator = pipeline('text-generation', model=model_name)
    set_seed(42)
    prompt = f"""
    # Based on the following detailed research plan, generate a Python script that outlines the necessary steps for conducting the experiment. The script should cover data loading, preprocessing, model training (if applicable), and evaluation, incorporating the specified tools, datasets, and evaluation metrics from the plan.
    
    # Research Plan:
    {research_plan}
    """
    generated_script = generator(prompt, max_length=1000, num_return_sequences=1)[0]['generated_text']
    print(generated_script) 
    return generated_script


def save_and_execute_script(script, filename="experiment.py"):
    with open(filename, 'w') as file:
        file.write(script)
    subprocess.run(["python", filename], check=True)



paper = search_arxiv(TOPIC)
rq = generate_research_question(paper)
rp = generate_research_plan(rq)
experiment_script = generate_experiment(rq)
print(f"Paper and Summary: {paper}\n\n\nResearch Question: {rq}\n\n\n Research Plan: {rp}\n\n\nExperiment Script:{experiment_script}")

# save_and_execute_script(experiment_script)