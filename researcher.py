#from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from xml.etree import ElementTree
import fitz  # PyMuPDF
import subprocess
import requests

import re

n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 4096  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

class Researcher:
    def __init__(self, model_path="vicgalle/Mixtral-7Bx2-truthy", code_model_path="Phind/Phind-CodeLlama-34B-v2", maxlen=1024):
        self.model_path = model_path
        self.maxlen = maxlen

        self.language_model = LlamaCpp(
            model_path="/path/to/model/model.gguf",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            n_ctx=4096,
            f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
            callback_manager=callback_manager,
            verbose=True,
        )

        #self.language_tokenizer = AutoTokenizer.from_pretrained(model_path)
        #self.language_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        #self.code_tokenizer = AutoTokenizer.from_pretrained(code_model_path)
        #self.code_model = AutoModelForCausalLM.from_pretrained(code_model_path, device_map='auto')
        #set_seed(42)

    def read_file_contents(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()

    def search_arxiv(self, keywords):
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
        #inputs = self.language_tokenizer(prompt, return_tensors="pt", max_length=self.maxlen, truncation=True)
        #outputs = self.language_model.generate(**inputs, max_length=self.maxlen, num_return_sequences=1)
        generated_text = self.language_model(prompt)
        return generated_text
    
    def generate_code(self, prompt):
        inputs = self.code_tokenizer(prompt, return_tensors="pt", max_length=self.maxlen, truncation=True)
        outputs = self.code_model.generate(**inputs, max_length=self.maxlen, num_return_sequences=1)
        generated_text = self.code_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def refine_research_goal(self, initial_goal):
        prompt = f"Refine this research goal based on existing literature: {initial_goal}" # Should retrieve some latest literature
        return self.generate_text(prompt)

    def generate_research_question(self, paper_text):
        prompt = f"Based on the following paper, generate research a new question to pursue:\n\n{paper_text}"
        return self.generate_text(prompt)

    def generate_research_plan(self, research_question):
        prompt = f"""
        Given the research question: "{research_question}", generate a detailed research plan. The plan should outline the objectives, methodologies, tools, datasets, and evaluation metrics.
        """
        return self.generate_text(prompt)

    def generate_experiment_script(self, research_plan):
        #        First think through the pieces of the code step by step to yourself, and then write them as comments in the code block as you generate the code.
        prompt = f"""

        You are an expert Python developer who writes perfect Python scripts from a given research plan.

        Based on the following research plan:

        {research_plan}

        Generate a Python script to conduct the experiment and display the findings. 
        Return a block of valid executable Python code wrapped in <code> tags and nothing else in your response, only code:

        """

        return self.generate_text(prompt)

    def process_paper(self, topic):
        paper_details = self.search_arxiv(topic)
        full_text = self.scrape_pdf_content(paper_details['pdf_link']) if 'pdf_link' in paper_details else paper_details['summary']
        refined_goal = self.refine_research_goal(full_text)
        research_question = self.generate_research_question(refined_goal)
        research_plan = self.generate_research_plan(research_question)
        experiment_script = self.generate_experiment_script(research_plan)

        print(f"Paper and Summary: {paper_details}\n\nRefined Goal: {refined_goal}\n\nResearch Question: {research_question}\n\nResearch Plan: {research_plan}\n\nExperiment Script: {experiment_script}")

    def save_and_execute_script(self, script, filename="experiment.py"):
        with open(filename, 'w') as file:
            file.write(script)
        try:
            result = subprocess.run(["python3", filename], capture_output=True, text=True, check=True)
            output_filename = filename.replace('.py', '_output.txt')
            with open(output_filename, 'w') as output_file:
                output_file.write(result.stdout)
            print(f"Results of {filename}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error executing {filename}:")
            print(e.stderr)  
            output_filename = filename.replace('.py', '_error.txt')
            with open(output_filename, 'w') as error_file:
                error_file.write(e.stderr)  # Save error output for examination - not used now 
            return None  # or raise e
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

def main(topic):
    researcher = Researcher()
    iterations = 3 

    research_questions = [] 
    research_plans = []
    experiment_scripts = []
    experiment_outputs = []

    paper_details = researcher.search_arxiv(topic)
    #if 'pdf_link' in paper_details:
        #full_text = researcher.scrape_pdf_content(paper_details['pdf_link'])
    #else:
    full_text = paper_details['summary']

    print(full_text)

    initial_research_question = researcher.generate_research_question(full_text)
    print(f"Topic:{topic}\n\n\nInitial RQ: {initial_research_question}")
    research_questions.append(initial_research_question)

    for iteration in range(iterations):
        print(f"Iteration {iteration+1} of {iterations}")

        if iteration > 0:
            refined_question = researcher.analyze_and_refine_question(research_plans, experiment_scripts, experiment_outputs, iteration - 1)
            research_question = refined_question
        else:
            research_question = initial_research_question

        research_plan = researcher.generate_research_plan(research_question)
        research_plans.append(research_plan)

        experiment_script = researcher.generate_experiment_script(research_plan)

        #reg_str = "<code>(.*?)</code>"
        #result = re.findall(reg_str, experiment_script)

        experiment_scripts.append(experiment_script)

        output_filename = researcher.save_and_execute_script(str(experiment_script))  
        experiment_outputs.append(output_filename)

    paper_sections = {
        "Introduction": "This paper explores the topic of " + topic + ".",
        "Literature Review": "\n\n".join(research_questions),
        "Methodology": "\n\n".join(research_plans),
        "Experiments and Results": "\n\n".join(experiment_scripts),
        "Discussion": "This section discusses the implications of the findings.",
        "Conclusion": "Here we conclude our findings and propose future work based on the experiments conducted."
    }

    research_paper = "\n\n".join([f"{section}:\n{content}" for section, content in paper_sections.items()])
    print("\nFinal Research Paper Draft:\n")
    print(research_paper)

    with open("research_paper.txt", "w") as file:
        file.write(research_paper)

if __name__ == "__main__":
    topic = "Sorting Algorithms"
    main(topic)
