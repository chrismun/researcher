from transformers import pipeline, set_seed

def generate_research_plan(research_question, model_name='EleutherAI/gpt-neo-2.7B'):
    """
    Generates a detailed research plan for the provided research question.
    
    Args:
    - research_question (str): The research question to base the plan on.
    - model_name (str): The model to use for generating the research plan.
    
    Returns:
    - str: A detailed research plan.
    """
    generator = pipeline('text-generation', model=model_name)
    set_seed(42)
    prompt = f"""
    # Given the research question: "{research_question}", generate a detailed research plan. The plan should outline the objectives, methodologies, tools (e.g., PyTorch, NAMD), datasets, and evaluation metrics to be used. Consider that the execution will be performed on a typical compute cluster with command-line capabilities.
    
    # Research Question: {research_question}
    """
    research_plan = generator(prompt, max_length=1000, num_return_sequences=1)[0]['generated_text']
    print(research_plan)
    return research_plan
