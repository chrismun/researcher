import subprocess
from transformers import pipeline, set_seed

def generate_experiment(research_question, model_name='EleutherAI/gpt-neo-2.7B'):
    generator = pipeline('text-generation', model=model_name)
    set_seed(42)
    prompt = f"""
    # Given the research question: "{research_question}", generate a Python script that outlines an experiment to investigate this question. The script should include data loading, preprocessing, model training (if applicable), and evaluation. Use placeholders for custom logic or data.
    
    # Research Question: {research_question}
    """
    generated_script = generator(prompt, max_length=1000, num_return_sequences=1)[0]['generated_text']
    return generated_script

research_question = "How does changing the learning rate affect the accuracy of a convolutional neural network on the MNIST dataset?"
experiment_script = generate_experiment(research_question)
print(experiment_script)

def save_and_execute_script(script, filename="experiment.py"):
    """
    Saves the generated script to a file and executes it.
    
    Args:
    - script (str): The Python script to execute.
    - filename (str): The name of the file to save the script to.
    """
    with open(filename, 'w') as file:
        file.write(script)
    
    subprocess.run(["python", filename], check=True)

save_and_execute_script(experiment_script)
