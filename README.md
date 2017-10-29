# text_classification_project
The project is aimed at implementing multiple machine learning methods on texts to classify and cluster news topics.

### Motivation
In read world practice, we always have a need to categorize received information. That is for the convenience of storing and retrieving. Text classification is widely researched topic. In our project, we attempt to dive deeper into the most popular machine learning methods used for text classification. For this midterm project, we chose multinomial Naïve Bayesian, Support Vector Machine, and Random Forest for implementation, and compare their performance in text classification.

### Data Preparation
The dataset used in this project is news articles published by English Newspaper Guardian during 2012 to present. The Guardian created its own Open Source platform to all the contents the newspaper created, categorized by tags and section. Developers only need to register for an API key to get access to the whole dataset.

### Data Exploration
Statistics here.

### Feature Extraction
First, raw data are lowercased and tokenized by word. 
Second, each word is stemmed using Porter Stemmer.
third, calculate tfidf sparse vector for 1, 2, 3 grams for each article, with all stop_words erased, max_df (0.5), min_df(20).
Momery Issue, calculating tfidf matrix for 1, 2, 3 grams usually drastically inflates the size of the result. However, using forementioned steps of feature extraction reduced the size of resulted tfidf matrix down to a level a usual laptop can handle.

