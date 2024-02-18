# Movie Recommendation System

This project implements a content-based movie recommendation system using Python. It utilizes the TMDB 5000 Movie Dataset available on Kaggle.

## Dataset
- **Name:** [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)

## Overview
The recommendation system is built using a content-based approach, where movies are recommended based on their features such as genres, keywords, cast, and crew. The project is implemented in Python and leverages various libraries for data manipulation, text processing, and machine learning.

## Implementation Steps
### Data Collection:
- The TMDB 5000 Movie Dataset is downloaded from Kaggle. This dataset contains information about movies, including their titles, genres, keywords, cast, crew, and overview.

### Data Preprocessing:
- The dataset is loaded into pandas DataFrames for manipulation.
- Columns with relevant information for recommendations, such as movie title, overview, genres, keywords, cast, and crew, are selected.
- Missing values and duplicates are removed from the dataset using pandas functions.
- Strings representing lists of values (e.g., genres, keywords, cast) are converted into actual lists using the ast library.

### Feature Engineering:
- Text data in the overview, genres, keywords, cast, and crew columns are processed and concatenated into a single column called "tags".
- Text preprocessing techniques such as tokenization, lowercase conversion, and stemming (using NLTK's PorterStemmer) are applied to the "tags" column to normalize the text data.

### Vectorization:
- CountVectorizer from scikit-learn is used to convert the text data into numerical vectors.
- The vectorized data is used to compute cosine similarity between movies, forming the basis for recommendations.

### Recommendation Function:
- A function is implemented to recommend movies based on cosine similarity scores.
- Given a movie title as input, the function returns the top recommended movies based on similarity scores.

## Future Scope:
- The project can be extended to include a user interface using libraries like Streamlit.
- Deployment of the recommendation system as a web application on platforms like Heroku, AWS, or Azure.

## Library Installation
You can install the required libraries using pip:

```bash
pip install numpy pandas matplotlib nltk scikit-learn

## How to Use
1. Clone the repository to your local machine.
2. Download the TMDB 5000 Movie Dataset from the provided link.
3. Install the required libraries using the provided command.
4. Run the Python script to execute the movie recommendation system.
5. Explore the recommendations based on the input movie.

### Usage Example
```python
# Example usage of the recommendation system
recommend('Avatar')


