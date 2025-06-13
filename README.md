# Detecting most important words (terms) per news tag using Pyspark

News articles are often associated with tags, that allow to easier navigate the news portals as well as find all the relevant articles related to the topic. It is important to accurately classify the news, so that the reader would easily find all the relevant information. <br>
I decided to analyze what are the most important words per news article tag, so that this information could be used to later build tag recommender system or maybe facilitate other unsupervised learning tasks that involve article topic classification. Besides the practical benefit, this project helped to gain ingsights on different natural language processing (NLP) techniques, explore the meaning of "importance" in terms, as well as delve into the possibilities of Pyspark tools for NLP. <br>

**Data:** Medium articles <br>
**Big data technologies used:** Pyspark <br>
**Additional technologies:** NLTK <br>

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
Pyspark dataframe was used to store data. Using `sql.features` model from Pyspark, data was further cleaned by converting it to lower case and removing punctuation with regex. In this task the punctuation is not important, since the goal is to find the most importan words, it is not affected by punctuation. <br>
A common way to store data in NLP related tasks is to convert it to the list of tokens. This step was performed using Pyspark's `ml.feature` module.  In addition, this task requires the removal of stop words. Stop words are the class of words that appear very frequently. Since the idea of the project is to find the words that uniquely motivate the choice of a tag, common words like "the", "not", "could", etc., would negatively impact the results. This issue can be conveniently solved by supplying a list of common english stop words (source - NLTK library) to Pyspark's function `StopWordsRemover`. <br> 
Each row of the dataframe object now contains original text, tags, tokens from original and stop word-removed text. <br>

3. Generate data features <br>
To measure the importance of a word, it first has to be represented in the way that allows some comparison or ranking. In NLP context, string tokens are usually converted to numeric vectors. There are many ways to do that, but in order to also measure importance, two complexity levels will be used: word count and term frequency - inverse document frequency (TF-IDF). <br>

**Word count:** <br>
It is assumed that the most frequent word for one tag, throughout all the documents sharing this tag, could be the most representative. First, Pyspark's function from `ml.feature` module `CountVectorizer` is used to convert text to sparse vectors (vector represents the count of specific words). Then the counts are aggregated, so they would represent not the most frequent words per one text, but rather the most frequent words per all the texts that share this one tag. <br>

**TF-IDF**
TF-IDF lets to find the most representable words, meaning that the word, which appears often in one text and does not appear often in other text, gets the highest TF-IDF score. The words with highest scores then are deemed to be the most important. In order for the score to represent tags and not text, the results were also aggregated per tag. 

### Outcome
Comparing the most important words from the 5 most frequent tags, it is visible, that TFIDF finds very unique words, that usually are not overlapping with the most important words from other tags. Such outcome is expected, since this method finds very unique words. However, that might not be desirable in the pursuit of tag recommender, since we want the tags to be associated with many similar news articles, therefore such words like "sinx" or "setd" appearing in the article might not signal that the tag is "data science". On the other hand, the most important tags according to word frequency are more comprehensible, therefore the tag "data science" is associated with words "data", "AI" and "learning". <br>
However, while being less interpretable, TFIDF offers less overlapping important words per topic,  which might lead to more accurate tag suggestion. <br>
Additionally, it is useful to investigate the relationship between TFIDF scores and word frequencies from the NLP tools perspective. There is an inverse relationship between these two metrics - the word that is very frequent in tag has a low TFIDF score and vice versa - the word, which is less frequent, has very high TFIDF score. Which essentially is the core meaning of TFIDF. 