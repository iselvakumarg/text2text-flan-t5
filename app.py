from fastapi import FastAPI
from transformers import pipeline


## Create a new FASTAPI app instance
app = FastAPI()

## Initialize the text generation pipeline
pipe = pipeline("text2text-generation", model="google/flan-t5-small")


## Home Page
@app.get("/")
def home():
    return {"message": "Hello World!"}


@app.get("/generate")
def generate(text: str):
    """
    a function to handle the get request to generate text from a given input text
    """
    output = pipe(text)
    return {"output": output[0]['generated_text']}