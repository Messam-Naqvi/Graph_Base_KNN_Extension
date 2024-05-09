import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Function to scrape relevant text from a website link
def scrape_relevant_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract only relevant content (e.g., paragraphs)
    paragraphs = soup.find_all('p')
    relevant_text = ' '.join([p.get_text() for p in paragraphs])
    return relevant_text

# Function to summarize text
def summarize_text(text, max_words):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # Limit the number of words
    summarized_tokens = filtered_tokens[:max_words]
    return ' '.join(summarized_tokens)

# Function to preprocess the text
def preprocess_text(text):
    # Remove non-alphanumeric characters and lower the text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Remove words with length less than 3
    cleaned_tokens = [word for word in lemmatized_tokens if len(word) > 2]
    return cleaned_tokens

# Main function
def main():
    # URL of the website to scrape
    url = 'https://jamie.ideasasylum.com/2023/07/02/hobbies'
    # Scrape relevant text from the website
    relevant_text = scrape_relevant_text(url)
    # Summarize the text to 450 words
    summarized_text = summarize_text(relevant_text, 550)
    # Preprocess the text
    cleaned_tokens = preprocess_text(summarized_text)
    # Join tokens to form cleaned text
    cleaned_text = ' '.join(cleaned_tokens)
    # Calculate word count
    word_count = len(cleaned_tokens)
    print(f"Cleaned text:\n{cleaned_text}\n")
    print(f"Word count after preprocessing: {word_count}")

# Entry point of the script
if __name__ == "__main__":
    main()
