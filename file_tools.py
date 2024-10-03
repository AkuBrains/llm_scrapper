import requests
import io
import fitz
import PyPDF2
import os
import arxiv



def fetch_arxiv_articles(query, max_results=10, sort_by=arxiv.SortCriterion.Relevance):
    hidden_directory = '.pdfs'
    os.makedirs(hidden_directory, exist_ok=True)

    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_by
    )

    results = list(client.results(search))
    articles = []
    for result in results:
        filename = result.pdf_url.split('/')[-1]
        filename = f"{filename}.pdf"
        file_path = os.path.join(hidden_directory, filename)
        if not os.path.exists(file_path):
            try:
                result.download_pdf(dirpath=hidden_directory, filename=filename)
            except Exception as e:
                print(e)
                continue
        articles.append((result.title, file_path))
    return articles


def fetch_pdf_files(directory):
    # List to store paths and filenames of PDFs
    articles = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file is a PDF
            if file.endswith('.pdf'):
                # Create full path to the file
                full_path = os.path.join(root, file)
                articles.append((file, full_path))

    return articles


def extract_text_from_online_pdf(url):
    response = requests.get(url)
    pdf = io.BytesIO(response.content)
    with fitz.open(stream=pdf) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text


def extract_text_from_pdf(file_path):
    # Open the PDF file
    with open(file_path, 'rb') as pdf_file:
        # Initialize the PDF reader
        reader = PyPDF2.PdfReader(pdf_file)

        # Initialize a variable to store extracted text
        text = ""

        # Loop through each page in the PDF
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

    return text


def aggregate_files(file_names, output_file):
    with open(output_file, 'w') as outfile:
        for file_name in file_names:
            with open(file_name, 'r') as infile:
                # Write the content of each file to the final output file
                outfile.write(infile.read())
        print(f"All task outputs aggregated into {output_file}")


