

Hello! This file contains information about the github repository. The repository is about AI-powered Medical Literature Review using Large Language Models. 

It contains codes for the first screening, full text screening using Retrieval Augmentd Generation Pipeline and codes for web scrappers to automatically extract titles and abstrcts and download PDFS. 

### IMPORTANT ###

This project utilizes Large Language Models from Ollama SO make sure you have Ollama installed and your desired models pulled already. 

For list of Ollama models go to: (https://ollama.com/library)


### First Screening ###

1. first_screening.ipynb : Code for the title and abstract screening of papers with three models. 

2. first_screening_inference.ipynb : Code to check the outputs, convert them into 0 1 classification labels based on the criteria and make inferences based on the confusion matrix and classification reports. 

3. first_screening_outputs : Stores the outputs for all the first screening experiemnts. 


### Full-Text Screening ###

1. full_text_screening.ipynb : Code for the full text screening experiments using a RAG based pipeline. It also contains the Confusion matrix and classification reports.

2. rag_llama.py : Class definitaion of the RAG based pipeline. The components are coded as functions which can be called one by one while execution of the pipeline. 

3. full_text_screening_outputs: Stores the outputs for both the full text screening experiemnts. 


### Web Scrappers ###

1. scrapper_pubmed.py : A seleinum based webscrapper that takes the citation of the paper as an inout and searches for the paper on the github website. Then it extracts the Title, Abstract and DOI for that paper. 

2. scrapper_scihub.py : A selenium based webscrapper that takes a DOI as an input and looks for the paper on Scihub. It then downloads the paper to the specified download directory. 

3. scrapper_pubMed.ipynb : This notebook takes in the UVH first screening results as the ground truth. Runs the scrapper_pubmed first to extract the Titles, Abstracts and DOI. Then runs the scrapper_scihub to download all the required papers. The UVH- Screening (First stage)(ground truth).xlsx contains a lot of papers listes but this project only focuses on the files from source PUBMED. 