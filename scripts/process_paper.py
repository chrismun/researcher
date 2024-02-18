from transformers import pipeline, set_seed
from xml.etree import ElementTree
import subprocess
import requests
import fitz

topic="OpenACC Validation and Verification"
# model_path="yunconglong/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B"
model_path="vicgalle/Mixtral-7Bx2-truthy"
maxlen = 1024

def search_arxiv(keywords):
    query = '+AND+'.join(keywords.split())
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=1'
    response = requests.get(url)
    root = ElementTree.fromstring(response.content)
    paper_details = {}

    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        paper_details['title'] = entry.find('{http://www.w3.org/2005/Atom}title').text
        paper_details['summary'] = entry.find('{http://www.w3.org/2005/Atom}summary').text
        for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
            if link.attrib.get('title') == 'pdf':
                paper_details['pdf_link'] = link.attrib['href'] + ".pdf"
                break

    return paper_details

def scrape_pdf_content(pdf_url):
    response = requests.get(pdf_url)
    with open("temp_paper.pdf", "wb") as f:
        f.write(response.content)

    doc = fitz.open("temp_paper.pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    return text

def generate_research_question(paper_text, model_name=model_path, num_questions=1):
    generator = pipeline('text-generation', model=model_name, device_map='auto')
    set_seed(42)
    prompt = f"Based on the following paper, generate {num_questions} research question:\n\n{paper_text}\n\nQuestions:"
    questions = generator(prompt, max_length=maxlen, num_return_sequences=1)
    return questions[0]['generated_text']

def generate_research_plan(research_question, model_name=model_path):
    generator = pipeline('text-generation', model=model_name, device_map='auto')
    set_seed(42)
    prompt = f"""
    # Given the research question: "{research_question}", generate a detailed research plan. The plan should outline the objectives, methodologies, tools (e.g., PyTorch, NAMD), datasets, and evaluation metrics to be used. Consider that the execution will be performed on a typical compute cluster with command-line capabilities.
    
    # Research Question: {research_question}
    """
    research_plan = generator(prompt, max_length=maxlen, num_return_sequences=1)[0]['generated_text']
    print(research_plan)
    return research_plan


def generate_experiment(research_plan, model_name=model_path):
    generator = pipeline('text-generation', model=model_name, device_map='auto')
    set_seed(42)
    prompt = f"""
    # Based on the following research plan, generate a Python script that implements the necessary steps for conducting the experiment. The script should cover data loading, preprocessing, model training (if applicable), and evaluation, incorporating the specified tools, datasets, and evaluation metrics from the plan.
    
    # Research Plan:
    {research_plan}
    """
    generated_script = generator(prompt, max_length=maxlen, num_return_sequences=1)[0]['generated_text']
    print(generated_script) 
    return generated_script

def save_and_execute_script(script, filename="experiment.py"):
    with open(filename, 'w') as file:
        file.write(script)
    subprocess.run(["python", filename], check=True)

def process_paper(topic):
    paper_details = search_arxiv(topic)
    if 'pdf_link' in paper_details:
        full_text = scrape_pdf_content(paper_details['pdf_link'])
    else:
        full_text = paper_details['summary']
    research_question = generate_research_question(full_text)
    research_plan = generate_research_plan(research_question)
    experiment_script = generate_experiment(research_plan)
    print(f"Paper and Summary: {paper_details}\n\n\nResearch Question: {research_question}\n\n\n Research Plan: {research_plan}\n\n\nExperiment Script:{experiment_script}")


process_paper(topic)

# save_and_execute_script(experiment_script)