import datetime

from file_tools import fetch_arxiv_articles
from llm_tools import run_single_summarizer, GeminiFactory, SeqGeminiFactory, configure_gemini
from file_tools import extract_text_from_pdf

NUM_PROCESS = 1

CUSTOM_PROMPT = {
    "RL researcher":f"You are a researcher studying cooperative AI and multi-agent reinforcement learning. "
                 f"You are currently making the state of the art. For each article, you need to write a summary with the following format : "
                 f"- The context : describe the problems addressed in the article"
                 f"- The contributions : summarize what have been done in the article and describe with details the solution"
                 f"- The experiments :  summarize the results of the experiments and highlight the take-away of the experiments"
                 f"- The limits : eventually describe the limits of their solution"
                 f"- The next steps"
                 f"- Conclusion",
    "SEQ":(f'You are a researcher studying cooperative AI and multi-agent reinforcement learning.\n'
           f'A student will ask questions about an article. You just have to answer his question without any surrounding text.\n',
           [ ('Context', f'What is the context ?'),
             ('Contributions and related work', f'What are the contributions ? What was done in the article to tackle the problem ?'),
             ('Experiments', f'What are the experiments ?'),
             ('Limits', f'What are the limits of their solution or work?'),
             ('Next steps', f'What are the next steps of their work?')])

}


if __name__ == '__main__':
    configure_gemini("your_api_key")

    max_result = 10
    theme = "mirror reinforcement learning"

    # Define the year you want
    start_year = 2022
    end_year = 2025

    # Create a datetime object for January 1st, 00:00:00 of that year
    sdt = datetime.datetime(start_year, 1, 1)
    edt = datetime.datetime(end_year, 1, 1)

    # ti:Multi agent OR ti:cooperative AND
    query = f"abs:{theme} AND submittedDate:[{sdt.strftime('%Y-%m-%d')} TO {edt.strftime('%Y-%m-%d')}]"

    articles = fetch_arxiv_articles(query, max_results=max_result)
    print("Article count:", len(articles))

    run_single_summarizer(extract_text_from_pdf,
                          SeqGeminiFactory,
                          articles,
                          output_file=f"{theme.replace(' ','_')}_{max_result}_arxiv.md",
                          prompt=CUSTOM_PROMPT["SEQ"][0],
                          seq_instruct=CUSTOM_PROMPT["SEQ"][1])
