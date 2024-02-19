from transformers import AutoTokenizer, set_seed
from xml.etree import ElementTree
from termcolor import colored
import fitz  # PyMuPDF
import subprocess
import requests
import re

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class Researcher:
    def __init__(self, model_path="/path/to/model/model.gguf", code_model_path="Phind/Phind-CodeLlama-34B-v2", maxlen=8192, n_gpu_layers=1, n_batch=4096):
        self.model_path = model_path
        self.maxlen = maxlen
        self.code_model_path = code_model_path
        self.language_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_model_path)
        set_seed(42)
        
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
        self.language_model = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=self.maxlen,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=False,
            echo=False
        )

        self.code_model = LlamaCpp(
            model_path=code_model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=self.maxlen,
            f16_kv=True,
            callback_manager=callback_manager,
            verbose=False,
            echo=False
        )

    def read_file_contents(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()

    def search_arxiv(self, keywords):
        query = '+AND+'.join(keywords.split())
        url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=10'  
        response = requests.get(url)
        root = ElementTree.fromstring(response.content)
        papers_details = []

        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            paper_detail = {
                'title': entry.find('{http://www.w3.org/2005/Atom}title').text,
                'summary': entry.find('{http://www.w3.org/2005/Atom}summary').text,
                'pdf_link': None 
            }
            for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
                if link.attrib.get('rel') == 'alternate' and link.attrib['type'] == 'text/html':
                    paper_detail['pdf_link'] = link.attrib['href'].replace('abs', 'pdf') + ".pdf"  
            papers_details.append(paper_detail)

        return papers_details

    def scrape_pdf_content(self, pdf_url):
        response = requests.get(pdf_url)
        with open("temp_paper.pdf", "wb") as f:
            f.write(response.content)

        doc = fitz.open("temp_paper.pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        return text

    def generate_text(self, prompt):
        inputs = self.language_tokenizer(prompt, return_tensors="pt", max_length=self.maxlen, truncation=True)
        # Move input_ids to the same device as the model
        inputs = inputs.to(self.language_model.device)
        outputs = self.language_model.generate(**inputs, max_length=self.maxlen + 1024, num_return_sequences=1)
        generated_text = self.language_tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_text = generated_text.replace(prompt, "").strip()
        return cleaned_text


    def generate_one_completion(self, prompt: str):
        self.code_tokenizer.pad_token = self.code_tokenizer.eos_token
        inputs = self.code_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = inputs.to(self.code_model.device)

        generate_ids = self.code_model.generate(inputs.input_ids, max_new_tokens=1024, do_sample=True, top_p=0.75, top_k=40, temperature=0.1)
        completion = self.code_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        completion = completion.replace(prompt, "").split("\n\n\n")[0]

        return completion



    def refine_research_goal(self, initial_goal):
        prompt = f"Refine this research goal based on existing literature: {initial_goal}"
        return self.generate_text(prompt)

    def generate_research_question(self, paper_text):
        prompt = f"Based on the following paper, generate a new research question to pursue. Make a question that is testable via computation, with machine learning or simulation or something else.\n\n{paper_text}"
        return self.generate_text(prompt)

    def generate_research_plan(self, research_question):
        prompt = f"""
Given the research question: "{research_question}", generate a detailed research plan. The plan should outline the objectives, methodologies, tools, datasets, and evaluation metrics.
"""
        return self.generate_text(prompt)

    def generate_experiment_plan(self, research_plan):
        prompt = f"""
Based on the following research plan, generate a detailed plan for conducting the experiment. The plan should outline specific objectives, methodologies, tools, datasets to be used, expected outcomes, and evaluation metrics. It should be a plan for a python script, using ML or simulation or other techniques.\n\n
{research_plan}
"""
        return self.generate_text(prompt)

    def generate_experiment_script(self, research_plan):
        prompt = f"""
Based on the following experiment plan, generate a Python script to conduct the experiment. Note that you dont have access to any previous data unless you explicitly retrieve it. Output the findings. Surround the code in backticks.\n\n
{research_plan}

IMPORTANT: Generate your own data, there are no csv files available.
"""
        print(prompt)
        return self.generate_one_completion(prompt)

    def process_paper(self, topic):
        paper_details = self.search_arxiv(topic)
        full_text = self.scrape_pdf_content(paper_details['pdf_link']) if 'pdf_link' in paper_details else paper_details['summary']
        refined_goal = self.refine_research_goal(full_text)
        research_question = self.generate_research_question(refined_goal)
        research_plan = self.generate_research_plan(research_question)
        experiment_script = self.generate_experiment_script(research_plan)

        print(f"Paper and Summary: {paper_details}\n\nRefined Goal: {refined_goal}\n\nResearch Question: {research_question}\n\nResearch Plan: {research_plan}\n\nExperiment Script: {experiment_script}")


    def save_and_execute_script(self, script, filename="experiment.py"):
        code_pattern = r"```python(.*?)```"
        code_match = re.search(code_pattern, script, re.DOTALL)
        
        if code_match:
            code_to_execute = code_match.group(1)
            lines = code_to_execute.split('\n')
            stripped_lines = [line.strip() for line in lines]
            code_to_execute = '\n'.join(stripped_lines)
        else:
            print("No code found within backticks.")
            return None

        with open(filename, 'w') as file:
            file.write(code_to_execute)

        try:
            result = subprocess.run(["python", filename], capture_output=True, text=True, check=True)
            output_filename = filename.replace('.py', '_output.txt')
            with open(output_filename, 'w') as output_file:
                output_file.write(result.stdout)
            print(f"##### Output saved at {output_filename}")
            
        except subprocess.CalledProcessError as e:
            formatted_error = f"```\n{e.stderr}\n```"
            print(formatted_error)
            
            output_filename = filename.replace('.py', '_error.txt')
            with open(output_filename, 'w') as error_file:
                error_file.write(formatted_error)
            
            return None
        return output_filename

    def analyze_and_refine_question(self, research_plans, experiment_scripts, experiment_outputs, iteration):
        research_plan_content = research_plans[iteration]
        experiment_script_content = experiment_scripts[iteration]
        output_content = experiment_outputs[iteration]

        prompt = (
            f"Research Plan:\n{research_plan_content}\n\n"
            f"Experiment Script:\n{experiment_script_content}\n\n"
            f"Experiment Output:\n{output_content}\n\n"
            "Based on the research plan, the experiment script, and the output of the experiment, "
            "how should the research question be refined to better align with the observed results and insights?"
        )

        refined_question = self.generate_text(prompt)
        return refined_question


def process_single_paper(researcher, paper_detail):
    print(colored(f"Processing paper: {paper_detail['title']}", 'yellow'))
    full_text = researcher.scrape_pdf_content(paper_detail['pdf_link']) if 'pdf_link' in paper_detail and False else paper_detail['summary']
    
    iterations = 3
    research_questions, research_plans, experiment_scripts, experiment_outputs = [], [], [], []

    for iteration in range(iterations):
        if iteration == 0:
            research_question = researcher.generate_research_question(full_text)
        else:
            research_question = researcher.analyze_and_refine_question(research_plans, experiment_scripts, experiment_outputs, iteration - 1)
        
        print(colored(f"Iteration {iteration+1}: Research Question - {research_question}", 'green'))

        research_plan = researcher.generate_research_plan(research_question)
        print(colored(f"Research Plan: {research_plan}", 'blue'))
        
        experiment_plan = researcher.generate_experiment_plan(research_plan)
        print(colored(f"Experiment Plan: {experiment_plan}", 'yellow'))

        experiment_script = researcher.generate_experiment_script(experiment_plan)
        print(colored(f"Experiment Script: {experiment_script}", 'red'))
        
        output_filename = researcher.save_and_execute_script(experiment_script)

        research_questions.append(research_question)
        research_plans.append(research_plan)
        experiment_scripts.append(experiment_script)
        experiment_outputs.append(output_filename)

    paper_sections = {
        "Introduction": f"This paper explores findings related to: {paper_detail['title']}.",
        "Literature Review": "\n\n".join(research_questions),
        "Methodology": "\n\n".join(research_plans),
        "Experiments and Results": "\n\n".join(experiment_scripts),
        "Discussion": "This section discusses the implications of our findings.",
        "Conclusion": "We conclude our findings and propose future work."
    }

    research_paper = "\n\n".join([f"{section}:\n{content}" for section, content in paper_sections.items()])
    paper_title = paper_detail['title'].replace(' ', '_').replace('/', '_').replace(':', '_')
    filename = f"research_paper_{paper_title}.txt"
    with open(filename, "w") as file:
        file.write(research_paper)
    print(colored(f"Research paper for '{paper_detail['title']}' saved as '{filename}'", 'cyan'))

def main(topic):
    researcher = Researcher()
    papers_details = researcher.search_arxiv(topic)

    for paper_detail in papers_details:
        process_single_paper(researcher, paper_detail)

if __name__ == "__main__":
    topic = "OpenACC Validation and Verification"
    main(topic)