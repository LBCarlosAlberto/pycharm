# text_classification_project
The project is aimed at implementing multiple machine learning methods on texts to classify and cluster news topics.

### Problem Statement
In read world practice, we always have a need to categorize received information. That is for the convenience of storing and retrieving. Text classification is widely researched topic. In our project, we attempt to dive deep into the most popular machine learning methods used for text classification. For this midterm project, we chose multinomial Naïve Bayesian, Support Vector Machine, and Random Forest for implementation, and compare their performance in text classification.


### Data Preparation
The dataset used in this project is news articles published by English Newspaper Guardian during 2012 to present. The Guardian created its own Open Source platform to all the contents the newspaper created, categorized by tags and section. Developers only need to register for an API key to get access to the whole dataset. We used the API and downloaded a total of 236,378 pieces of news articles under nine sections: Business, Environment, Fashion, Life and Style, Politics, Sport, Technology, Travel, and World.

### Data Exploration
The downloaded data are in the form of json with multiple attributes, in which we only used the ‘section’, the topic name, and ‘bodyText’, the article content, for this project. 


By primary exploring the dataset, we find the articles have imbalanced distribution across all topics as shown in Figure 1. The topic ‘World’ covers the largest chunk, having 53,712 articles, while topics ‘fashion’ and ‘travel’ only have 7,413 and 6,528 articles respectively. 

### Feature Extraction
In this project, we decide to use each word’s ‘tf-idf’ statistic as a feature for model training. More specifically, we first build the total vocabulary in the whole dataset, and convert each article into a vocabulary-long ‘tf-idf’ vector. Machine learning models are trained using such features.
In practice, due to the large set of data (1 GB) and limited memory space. We need to reduce the size of the feature matrix. In order to achieve this goal, we first lowercased and tokenized all words, and then stemmed them using Porter Stemmer. While extracting ‘tfidf’ features, we erased all stop words, and filtered out words that appear in more than half of the articles or less than 20 articles.
Besides extracting ‘tf-idf’ feature matrix for 1-gram, we also extracted that for 2-gram, although we did not have enough time to train models on 2-gram matrix due to the tight midterm project deadline.

### Training Models
After we obtained ‘tf-idf’ feature matrix, we split the data into train (4/5) and test datasets (1/5). We use the train dataset for training, and test dataset for final evaluation and model comparison only. In this project up to midterm, we adopted three popular classification models: Multinomial Naïve Bayes, Support Vector Machine (SVM), and Random Forest.
In order to find optimal hyper parameters for each model, we used grid search cross validation method to exhaustively train models with all possible combinations of hyper parameters we predefined. 
For Multinomial Naïve Bayes, we only have one hyper parameter, which is the smoothing factor. We prepared a candidate set for the smoothing factor, [0.1, 0.5, 1, 2]. As a result, the model with smoothing factor of 0.1 is tested as optimal.
For Random Forest, we attempt to find the optimal model by checking following hyper parameters. In order to avoid over fitting, we checked the depth of a random tree, the size of leaves, and the threshold for splitting a tree node. We also need to check different evaluation criteria for tree performance such as ‘gini’ and ‘entropy’. Furthermore, in order to address the issue of data imbalance, we also tried different methods for weighting classes within the dataset. Below is the candidate sets we haved used in the grid search cross validation.
	{"max_depth": [3, None, 5, 10],
              "max_features": ['auto', 'sqrt', 'log2', 100],
              "min_samples_split": [10, 20],
              "min_samples_leaf": [3, 10, 20],
              'class_weight':['balanced', None, 'balanced_subsample'],
"criterion": ["gini", "entropy"]}
As a result, the optimal model is shown as follow.
RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample', criterion='gini', max_depth=10, max_features='auto', min_impurity_split=None, min_samples_leaf=3, min_samples_split=10, n_estimators=1000, oob_score=False, random_state=8)
For SVM, due to time issue, we have not yet obtained substantial result. 

### Evaluation
Figure 2 and Figure 3 present the evaluation results of Random Forest and MultinomialNB on the same testing data, respectively. 
As the results have shown, Multinomial Naïve Bayes perform better than Random Forest overall. Although both models do not achieve high precision and recall scores in predicting ‘Fashion’ and ‘Travel’, the two topics with the least data, the Naïve Bayes model still works better than the Random Forest. 

### Chanllenges in Training
Implementing a machine learning algorithm on a small dataset is one thing, but implementing the same algorithm on big data is a totally different story. In our practice, our training data is big enough to force us to take into account the time efficiency and memory space while we apply machine learning methods to the problem of interest.
Furthermore, the issue of data imbalanced distribution also raises our concern with the performance of our trained models. Further researches are also needed to find a better solution to this issue.

