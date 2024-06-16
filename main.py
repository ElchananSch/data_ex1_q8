import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import re


def clean_text(text):
    """Remove reference tags and extra newlines from the text."""
    text = re.sub(r'\[.*?\]+', '', text)  # Remove reference tags
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    return text

def fetch_page_content(title):
    """Fetch the content of a Wikipedia page using the API."""
    url = f"https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'titles': title,
        'prop': 'extracts|pageprops',
        'explaintext': True,
        'format': 'json',
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def get_page_text(page):
    """Extract the text from the page content."""
    pages = page['query']['pages']
    for page_id, page_data in pages.items():
        if 'extract' in page_data:
            return clean_text(page_data['extract'])
    return None

def is_disambiguation_page(page):
    """Check if the page is a disambiguation page."""
    pages = page['query']['pages']
    for page_id, page_data in pages.items():
        if 'pageprops' in page_data and 'disambiguation' in page_data['pageprops']:
            return True
    return False

def get_first_link_from_disambiguation(soup):
    """Get the first link from the disambiguation page that likely leads to the fruit page."""
    for link in soup.select('ul li a'):
        if 'fruit' in link.get_text().lower():
            href = link.get('href')
            last_slash_index = href.rfind('/')
            updated_fruit = href[last_slash_index + 1:]

            return updated_fruit.replace('_', ' ')
    return None

def get_wikipedia_text(fruit):
    # Fetch the initial page content using the Wikipedia API
    page = fetch_page_content(fruit)

    if not page:
        return None

    # Check if it's a disambiguation page
    if is_disambiguation_page(page):
        # Fetch the HTML content directly
        url = f"https://en.wikipedia.org/wiki/{fruit}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Find the first relevant link
            updated_fruit = get_first_link_from_disambiguation(soup)
            if updated_fruit:
                # Fetch and extract text from the specific fruit page
                return get_wikipedia_text(updated_fruit)
        return "Disambiguation page found but no specific fruit page link."
    else:
        return get_page_text(page)


def fruitcrawl(fruits):

    # Dictionary to store the text for each fruit
    fruit_texts = {}

    # Crawl each Wikipedia page and get the text
    for fruit in fruits:
        text = get_wikipedia_text(fruit)
        if text:
            fruit_texts[fruit] = text
        else:
            fruit_texts[fruit] = "Failed to retrieve text"

    # Save the text into a JSON file
    with open('fruit_texts.json', 'w', encoding='utf-8') as f:
        json.dump(fruit_texts, f, ensure_ascii=False, indent=4)


# Run the fruitcrawl function
def main():
    data = pd.read_csv('fruits.csv')
    fruits_list = data['Fruit'].tolist()
    fruitcrawl(fruits_list)


if __name__ == "__main__":
    main()

