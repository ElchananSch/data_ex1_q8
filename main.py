import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


################################### TF-IDF ##########################################
def unique_words(sentences_list):
    # Initialize an empty set to store unique words
    unique_words = set()

    # Iterate through each inner list and add each word to the set
    for sentence in sentences_list:
        words = word_tokenize(sentence)
        for word in words:
            unique_words.add(word)

    return list(unique_words)


def preprocess_text(text):
    # Remove non-alphanumeric characters (including punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert text to lowercase and split into words
    return text.lower().split()


def compute_tf(sentence, word_list):
    word_count = {}
    sentence = word_tokenize(sentence)
    for word in sentence:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1

    tf = {}
    for word in word_list:
        tf[word] = word_count.get(word, 0)
    return tf


def compute_df(sentences, word_list):
    df = {word: 0 for word in word_list}
    for sentence in sentences:
        sentence = word_tokenize(sentence)
        unique_words = set(sentence)
        for word in word_list:
            if word in unique_words:
                df[word] += 1
    return df


def compute_idf(df, total_documents):
    idf = {}
    for word, count in df.items():
        idf[word] = np.log((1 + total_documents) / (1 + count)) + 1
    return idf


def tfidf(sentences_list, words):
    # Preprocess descriptions
    # df['Description'] = df['Description'].apply(preprocess_text)
    #
    # # DF
    # descriptions = df['Description'].tolist()
    df_word = compute_df(sentences_list, words)

    # IDF
    total_documents = len(sentences_list)
    idf_word = compute_idf(df_word, total_documents)

    # TF and TF-ID
    tfidf_values = []
    for sentence in sentences_list:
        tf = compute_tf(sentence, words)
        tfidf_t = {word: tf[word] * idf_word[word] for word in words}
        tfidf_values.append(tfidf_t)

    # Create DataFrame
    tfidf_df = pd.DataFrame(tfidf_values)
    # tfidf_df.index = df.index

    return tfidf_df


########################### CRAWL WIKIPEDIA ###############################################
def clean_text(text):
    """Remove reference tags, specified characters, and extra newlines from the text."""
    # Remove reference tags in the form [reference]
    text = re.sub(r'\[.*?\]', '', text)
    # Remove specified characters: . - ,
    text = re.sub(r'[.,\-]', '', text)
    # Replace newlines with spaces
    text = text.replace('\n', ' ')
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


##################################### SUMMARIZTION WITH PAGE RANK ###########################


def remove_see_also_section(text):
    # Define the regex pattern to match "== See also ==" and everything that follows
    pattern = r"== See also ==.*"

    # Use re.sub to replace the matched pattern with an empty string
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)

    return cleaned_text


def textsum(json_file_path):
    # Load the JSON file

    nltk.download('punkt')
    nltk.download('stopwords')

    with open(json_file_path, 'r', encoding='utf-8') as file:
        fruit_data = json.load(file)

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    summaries = {}

    for fruit, text in fruit_data.items():
        # Tokenize text into sentences

        text = remove_see_also_section(text)
        sentences = sent_tokenize(text)
        # Preprocess each sentence
        processed_sentences = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            words = [stemmer.stem(word.lower()) for word in words if word.isalnum() and word.lower() not in stop_words]
            processed_sentences.append(' '.join(words))

        unique_words_fruit = unique_words(processed_sentences)
        tfidf_matrix = tfidf(processed_sentences, unique_words_fruit).to_numpy()

        # Compute similarity matrix (cosine similarity)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        zero_sum_cols = similarity_matrix.sum(axis=0) == 0

        # Normalize similarity_matrix
        normalized_data = similarity_matrix / similarity_matrix.sum(axis=0, keepdims=True)
        normalized_data[:, zero_sum_cols] = similarity_matrix[:, zero_sum_cols]  # Avoid division by zero
        similarity_matrix = normalized_data

        # Initialize PageRank scores
        beta = 0.85
        epsilon = 1e-5  # Convergence threshold
        M = similarity_matrix.copy()
        size_of_M = M.shape[0]
        r = np.full(size_of_M, 1 / size_of_M)
        r_new = np.zeros_like(r)
        max_iterations = 100  # Maximum number of iterations
        fixed_vector = np.full(size_of_M, (1 - beta) / size_of_M)

        # page rank calculation
        for i in range(max_iterations):
            r_new = beta * M @ r + fixed_vector
            if np.linalg.norm(r_new - r) < epsilon:
                break
            else:
                r = r_new

        ranked_indices = np.argsort(-r_new)

        # Extract summary (top 5 sentences)
        summary_sentences = [sentences[idx] for idx in ranked_indices[:5]]
        summary = ' '.join(summary_sentences)

        # Store the summary for the current fruit
        summaries[fruit] = summary

    return summaries

    # Print summaries
    # for fruit, summary in summaries.items():
    #     print(f"Summary for {fruit}:")
    #     print(summary)
    #     print()


def main():
    # section a - DONE
    data = pd.read_csv('fruits.csv')
    # fruits_list = data['Fruit'].tolist()
    # fruitcrawl(fruits_list)

    # section b - DONE
    summaries = textsum('fruit_texts.json')

    # section c - DONE

    summaries_df = pd.DataFrame(list(summaries.items()), columns=['Fruit', 'Summary'])

    summaries_list = list(summaries.values())
    for i in range(len(summaries_list)):
        summaries_list[i] = clean_text(summaries_list[i])
        summaries_list[i] = " ".join(preprocess_text(summaries_list[i]))

    unique_words_fruit = unique_words(summaries_list)
    tfidf_matrix_mine = tfidf(summaries_list, unique_words_fruit).to_numpy()  # Get feature names (words)

    # Function to get top n TF-IDF words for each fruit
    def get_top_3_words(tfidf_row):
        sorted_indices = np.argsort(-tfidf_row)
        top_3_indices = sorted_indices[:3]
        top_3_words = [(unique_words_fruit[idx], tfidf_row[idx]) for idx in top_3_indices]
        return top_3_words

    # Get top 3 words for each fruit
    top_words = {}
    for fruit, row in zip(summaries_df['Fruit'], tfidf_matrix_mine):
        top_words[fruit] = get_top_3_words(row)

    # Create a set of all unique top words
    unique_top_words = set(word for words in top_words.values() for word, score in words)

    # Initialize new columns with zeros
    for word in unique_top_words:
        data[word] = 0.0

    # Populate the new columns with the corresponding TF-IDF values
    for fruit, words in top_words.items():
        for word, score in words:
            data.loc[data['Fruit'] == fruit, word] = score

    pass

if __name__ == "__main__":
    main()
