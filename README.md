# Web Document Classification using Graph-Based Model

This project implements a document classification system using a graph-based extension of the k-Nearest Neighbors (kNN) algorithm. The project consists of two main Python scripts:

1. **Scrapping_Preprocessing.py**:
   - This script is responsible for web scraping data from online sources and preprocessing the scraped data for classification. It includes functions for fetching data from web pages, cleaning and formatting the data, and saving it to a usable format.

2. **Implementation.py**:
   - This script implements the graph-based kNN algorithm for document classification using the preprocessed data. It includes functions for training the classification model, performing predictions, and evaluating the results.

## Usage

1. Run `Scrapping_Preprocessing.py` to scrape data from web sources and preprocess it.
2. Run `Implementation.py` to train the classification model and perform document classification using the graph-based kNN algorithm.

## Data

- The `data` folder contains the training and testing datasets for document classification.
  - Training: 36 files
  - Testing: 9 files

## Dependencies

- Python 3.x
- Required Python libraries (e.g., BeautifulSoup for web scraping, networkx)

## Contributors

- Messam Naqvi
