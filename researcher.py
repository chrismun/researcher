from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from xml.etree import ElementTree
import fitz  # PyMuPDF
import subprocess
import requests

class Researcher:
    def __init__(self, model_path="vicgalle/gpt-neo-2.7B", maxlen=2048):
        self.model_path = model_path
        self.maxlen = maxlen
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
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
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=self.maxlen, truncation=True)
        outputs = self.model.generate(**inputs, max_length=self.maxlen, num_return_sequences=1)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def refine_research_goal(self, initial_goal):
        prompt = f"Refine this research goal based on existing literature: {initial_goal}" # Should retrieve some latest literature
        return self.generate_text(prompt)

    def generate_research_question(self, paper_text):
        prompt = f"Based on the following paper, generate research questions:\n\n{paper_text}"
        return self.generate_text(prompt)

    def generate_research_plan(self, research_question):
        prompt = f"""
        Given the research question: "{research_question}", generate a detailed research plan. The plan should outline the objectives, methodologies, tools, datasets, and evaluation metrics.
        """
        return self.generate_text(prompt)

    def generate_experiment_script(self, research_plan):
        prompt = f"""
        Based on the following research plan, generate a Python script for conducting the experiment:
        {research_plan}
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
        result = subprocess.run(["python", filename], capture_output=True, text=True, check=True)
        output_filename = filename.replace('.py', '_output.txt')
        with open(output_filename, 'w') as output_file:
            output_file.write(result.stdout)
        print(f"Results of {filename}:\n{result.stdout}")
        return output_filename

    def analyze_and_refine_question(self, research_plans, experiment_scripts, experiment_outputs, iteration):
        # Access the current iteration's research plan, experiment script, and output
        research_plan_content = research_plans[iteration]
        experiment_script_content = experiment_scripts[iteration]
        output_content = experiment_outputs[iteration]

        # Generate a prompt for the LLM that includes the contents and asks for refinement suggestions
        prompt = (
            f"Research Plan:\n{research_plan_content}\n\n"
            f"Experiment Script:\n{experiment_script_content}\n\n"
            f"Experiment Output:\n{output_content}\n\n"
            "Based on the research plan, the experiment script, and the output of the experiment, "
            "how should the research question be refined to better align with the observed results and insights?"
        )

        # Use the LLM to generate a refined research question or direction
        refined_question = self.generate_text(prompt)
        return refined_question


def main(topic):
    researcher = Researcher()
    iterations = 3 

    research_questions = [topic] 
    research_plans = []
    experiment_scripts = []
    experiment_outputs = []

    for iteration in range(iterations):
        print(f"Iteration {iteration+1} of {iterations}")

        if iteration == 0:
            research_question = researcher.generate_research_question(topic)
        else:
            refined_question = researcher.analyze_and_refine_question(research_plans, experiment_scripts, experiment_outputs, iteration - 1)
            research_question = refined_question

        research_questions.append(research_question)

        research_plan = researcher.generate_research_plan(research_question)
        research_plans.append(research_plan)

        experiment_script = researcher.generate_experiment_script(research_plan)
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