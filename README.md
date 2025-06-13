# Detecting most important words (terms) per news tag using Pyspark

News articles are often associated with tags, that allow to easier navigate the news portals as well as find all the relevant articles related to the topic. It is important to accurately classify the news, so that the reader would easily find all the relevant information. <br>
I decided to analyze what are the most important words per news article tag, so that this information could be used to later build tag recommender system or maybe facilitate other unsupervised learning tasks that involve article topic classification. Besides the practical benefit, this project helped to gain ingsights on different natural language processing (NLP) techniques, explore the meaning of "importance" in terms, as well as delve into the possibilities of Pyspark tools for NLP. <br>

*Data:* Medium articles <br>
*Big data technologies used:* Pyspark <br>
*Additional technologies:* NLTK <br>

### Solution
<!-- # Most important terms (words) per tag
# Useful for: Build a tag recommender; EDA of text data; comparison of common word embedding techniques - bag of words and tfidf
# Importance could be:
# Frequency
# TF-IDF
# Topic modelling using LDA (not sure yet) -->

1. Loading and preprocessing the data <br>
Text was cleaned by removing missing values, unnecessary columns and converting data types. For this analysis I used only article text and its tags. <br>
Since one article can have many tags, but essentially all the tags are related to the article, data was split, so that one article would have one tag. This results in many repetitions of the same article (as many as there are tags per article). <br>
This preprocessing allows to gain insights of the tag distribution: The most important tags (topics) of the set of articles were "Writing", "Artificial Intelligence" and "Creativity". <br>

2. NLP data cleaning methods using Pyspark <br>
Pyspark dataframe was used to store data. Using `sql.features` model from pyspark, data was further cleaned by converting it to lower case and removing punctuation with regex. In this task the punctuation is not important, since the goal is to find the most importan words, it is not affected by punctuation. <br>
A common way to store data in NLP related tasks is to convert it to the list of tokens. This step was performed using pyspark's `ml.feature` module.  In addition, this task requires the removal of stop words. Stop words are the class of words that appear very frequently. Since the idea of the project is to find the words that uniquely motivate the choice of a tag, common words like "the", "not", "could", etc., would negatively impact the results. This issue can be conveniently solved by supplying a list of common english stop words (source - NLTK library) to pyspark's function `StopWordsRemover`. <br> 
Each row of the dataframe object now contains original text, tags, tokens from original and stop word-removed text. <br>


### Outcome