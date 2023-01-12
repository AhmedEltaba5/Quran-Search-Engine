# Holy Quran Search Engine
## Description

A simple search engine through holy quran using NLP techniques for Arabic language. This approach uses TFIDF 
mainly to get the text similarity between the query and Quran text

![Holy Quran](https://github.com/AhmedEltaba5/Quran-Search-Engine/blob/main/img/quran.jpg?raw=true)

## Dataset

Arabic Holy Quran available via kaggle: https://www.kaggle.com/datasets/zusmani/the-holy-quran

Notebook available via kaggle: https://www.kaggle.com/code/ahmednassereltabakh/quran-search-engine/

## Result example

![search example](https://github.com/AhmedEltaba5/Quran-Search-Engine/blob/main/img/example.png?raw=true)

## How to Run 

```
git clone https://github.com/AhmedEltaba5/Quran-Search-Engine/
pip install -r requirements.txt
uvicorn app:app --reload
```
