from llm_tools import GradLlama31
import ollama

if __name__ == "__main__":
    with open('500_arxiv.md', 'r', encoding='utf-8') as file:
        content = file.read()

        pre_prompt1 = ("What are the main advances in Meta Learning ? Please, give the main contributions, limits and future works. Give me a list of articles dealing with Meta Learning.")

        pre_prompt = (
            "Here is the state of the art about cooperative artificial intelligence and multi agent reinforcement learning. The state of the art is written using MD files format and contains the summary of several articles. You have to sum up the key ideas by :\n "
            "- figuring out the main theme of research by sorting and rearranging the novelties\n"
            "- figuring out the principal challenges\n"    
            "- figuring out the future directions of research\n"
            "Please provide as much details as you can. The more you can, the better you are.")

        full_prompt = f"Here is the state of the art about cooperative ai and multi agent reinforcement learning: \n{content} \n\n {pre_prompt1}"


        response = ollama.chat(model='gradient-llama3.1-8b:latest',
                           messages=[{'role': 'user', 'content': full_prompt}],
                           stream=False)

        # response1 = ollama.chat(model='gradient-llama3.1-8b:latest',
        #                        messages=[{'role': 'user', 'content': "Add even more details"}],
        #                        stream=False)

    with open('SOTA.md', 'w', encoding='utf-8') as file:
        file.write(response["message"]["content"])
        file.close()

    print("Summary has been generated !")
