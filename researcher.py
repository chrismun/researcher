from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, LlamaForCausalLM
from xml.etree import ElementTree
from termcolor import colored
import fitz  # PyMuPDF
import subprocess
import requests
import re 

class Researcher:
    def __init__(self, model_path="meta-llama/Llama-2-13b-chat-hf", code_model_path="Phind/Phind-CodeLlama-34B-v2", maxlen=4098):
        self.model_path = model_path
        self.maxlen = maxlen
        self.language_tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.language_model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_model_path)
        self.code_model = LlamaForCausalLM.from_pretrained(code_model_path, device_map="auto")
        set_seed(42)

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
        inputs = self.language_tokenizer(prompt, return_tensors="pt", max_length=self.maxlen, truncation=True)
        outputs = self.language_model.generate(**inputs, max_length=self.maxlen + 1024, num_return_sequences=1)
        generated_text = self.language_tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_text = generated_text.replace(prompt, "").strip()
        return cleaned_text
    
    def generate_code(self, prompt):
        inputs = self.code_tokenizer(prompt, return_tensors="pt", max_length=self.maxlen, truncation=True)
        outputs = self.code_model.generate(**inputs, max_length=self.maxlen, num_return_sequences=1)
        generated_text = self.code_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def generate_one_completion(self, prompt: str):
        self.code_tokenizer.pad_token = self.code_tokenizer.eos_token
        inputs = self.code_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

        generate_ids = self.code_model.generate(inputs.input_ids.to("cuda"), max_new_tokens=1024, do_sample=True, top_p=0.75, top_k=40, temperature=0.1)
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

    def generate_experiment_script(self, research_plan):
        prompt = f"""
        Based on the following research plan, generate a Python script to conduct the experiment. Note that you dont have access to any previous data unless you explicitly retrieve it. Output the findings. Surround the code in backticks.\n\n
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

def main(topic):
    researcher = Researcher()
    iterations = 3 

    research_questions = [] 
    research_plans = []
    experiment_scripts = []
    experiment_outputs = []

    paper_details = researcher.search_arxiv(topic)
    if 'pdf_link' in paper_details and False:
        full_text = researcher.scrape_pdf_content(paper_details['pdf_link'])
    else:
        full_text = paper_details['summary']
    initial_research_question = researcher.generate_research_question(full_text)
    print(colored(f"##### Topic:{topic}\n\n\n##### Initial RQ: {initial_research_question}", "red"))
    research_questions.append(initial_research_question)

    for iteration in range(iterations):
        print(f"##### Iteration {iteration+1} of {iterations}")

        if iteration > 0:
            refined_question = researcher.analyze_and_refine_question(research_plans, experiment_scripts, experiment_outputs, iteration - 1)
            research_question = refined_question
        else:
            research_question = initial_research_question

        research_plan = researcher.generate_research_plan(research_question)
        print(colored(f"##### Research Plan: {research_plan}\n\n", "blue"))
        research_plans.append(research_plan)

        experiment_script = researcher.generate_experiment_script(research_plan)
        print(colored(F"##### Experiment Script: {experiment_script}", "green"))
        experiment_scripts.append(experiment_script)

        output_filename = researcher.save_and_execute_script(experiment_script)  
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
    topic = "OpenACC Validation and Verification"
    main(topic)
