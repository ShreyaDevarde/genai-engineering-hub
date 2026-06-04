import re
from dotenv import load_dotenv
from newspaper import Article

from langchain_groq import ChatGroq

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate
)

from langchain_core.output_parsers import JsonOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field


# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()


# ==========================================================
# Structured Output Schema
# ==========================================================

class NewsSummary(BaseModel):
    title: str = Field(description="Title of article")

    category: str = Field(
        description="Politics, Sports, Technology, Business, Health, Entertainment or World"
    )

    sentiment: str = Field(
        description="Positive, Negative or Neutral"
    )

    summary: list[str] = Field(
        description="3-5 concise bullet point summary"
    )

    key_people: list[str] = Field(
        description="Important people mentioned"
    )

    key_locations: list[str] = Field(
        description="Important locations mentioned"
    )


parser = JsonOutputParser(
    pydantic_object=NewsSummary
)


# ==========================================================
# Extract Article
# ==========================================================

def extract_article(url):

    article = Article(url)

    article.download()
    article.parse()

    return {
        "title": article.title,
        "text": article.text
    }


# ==========================================================
# Preprocessing
# ==========================================================

def preprocess(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# Split Long Articles
# ==========================================================

def split_article(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


# ==========================================================
# LLM
# ==========================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)


# ==========================================================
# Few-Shot Examples
# ==========================================================

examples = [

    {
        "article":
        """
        India launched a new weather satellite
        to improve forecasting and disaster
        preparedness.
        """,

        "summary":
        """
        {
            "title":"Weather Satellite Launch",
            "category":"Technology",
            "sentiment":"Positive",
            "summary":[
                "India launched a weather satellite",
                "Forecasting accuracy will improve",
                "Disaster preparedness is expected to improve"
            ],
            "key_people":[],
            "key_locations":["India"]
        }
        """
    },

    {
        "article":
        """
        RBI decided to keep interest rates unchanged
        due to inflation concerns.
        """,

        "summary":
        """
        {
            "title":"RBI Policy Decision",
            "category":"Business",
            "sentiment":"Neutral",
            "summary":[
                "RBI kept interest rates unchanged",
                "Inflation remains a concern",
                "Markets reacted cautiously"
            ],
            "key_people":[],
            "key_locations":["India"]
        }
        """
    }
]


# ==========================================================
# Example Prompt
# ==========================================================

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{article}"),
        ("ai", "{summary}")
    ]
)


few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt
)


# ==========================================================
# Main Prompt
# ==========================================================

prompt = ChatPromptTemplate.from_messages(

    [
        (
            "system",
            """
            You are an expert news analyst.

            Extract:

            1. Title
            2. Category
            3. Sentiment
            4. Summary (3-5 bullet points)
            5. Key People
            6. Key Locations

            Return ONLY valid JSON.

            {format_instructions}
            """
        ),

        few_shot_prompt,

        (
            "human",
            """
            Article:

            {article}
            """
        )
    ]
)


# ==========================================================
# Chain
# ==========================================================

chain = (
    prompt.partial(
        format_instructions=parser.get_format_instructions()
    )
    | llm
    | parser
)


# ==========================================================
# Summarization Function
# ==========================================================

def summarize_article(article_text):

    clean_text = preprocess(article_text)

    chunks = split_article(clean_text)

    combined_text = "\n\n".join(chunks)

    result = chain.invoke(
        {
            "article": combined_text
        }
    )

    return result


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    url = "https://indianexpress.com/article/explained/delhi-gymkhana-club-controversy-land-law-10708478/"

    article_data = extract_article(url)

    result = summarize_article(
        article_data["text"]
    )

    print("\nTITLE")
    print("-" * 50)
    print(result["title"])

    print("\nCATEGORY")
    print("-" * 50)
    print(result["category"])

    print("\nSENTIMENT")
    print("-" * 50)
    print(result["sentiment"])

    print("\nSUMMARY")
    print("-" * 50)

    for point in result["summary"]:
        print(f"• {point}")

    print("\nKEY PEOPLE")
    print("-" * 50)
    print(result["key_people"])

    print("\nKEY LOCATIONS")
    print("-" * 50)
    print(result["key_locations"])