import os
from dotenv import load_dotenv
from newspaper import Article
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

def extract_article(url):
    article = Article(url)
    article.download()
    article.parse()
    
    return {
        "title": article.title,
        "text": article.text
    }

def preprocess(text):
    text = text.replace("\n", " ")
    text = text.strip()
    return text[:4000]  

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_template("""
Summarize the following news article in 3-4 bullet points:{text}""")

chain = prompt | llm
def summarize_article(text):
    clean_text = preprocess(text)
    result = chain.invoke({"text": clean_text})
    return result.content

url = "https://indianexpress.com/article/explained/delhi-gymkhana-club-controversy-land-law-10708478/"

article_data = extract_article(url)

summary = summarize_article(article_data["text"])

print("TITLE:", article_data["title"])
print("\nSUMMARY:\n", summary)