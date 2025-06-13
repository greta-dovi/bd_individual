import os
import sys
import ast
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import ArrayType, StringType
import pyspark.ml.feature as ml
from pyspark.ml.functions import vector_to_array
import nltk
from nltk.corpus import stopwords
from pyspark.sql.window import Window

nltk.download('stopwords')

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# spark: SparkSession = SparkSession.builder\
#     .appName("individual")\
#     .config("spark.sql.autoBroadcastJoinThreshold", -1)\
#     .getOrCreate()

# Trying to limit resources
spark = SparkSession.builder \
    .appName("individual") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.autoBroadcastJoinThreshold", -1)\
    .config("spark.driver.maxResultSize", "2g") \
    .getOrCreate()

df = spark.read.option("header", "true")\
    .option("multiLine", "true")\
    .option("escape", '"')\
    .option("quote", '"')\
    .option("inferSchema", "true")\
    .csv("subset2.csv").cache()

# df = spark.read.option("header", "true")\
#     .option("multiLine", "true")\
#     .option("escape", '"')\
#     .option("quote", '"')\
#     .option("inferSchema", "true")\
#     .csv("medium_articles.csv")


# An individual assignment shall be done by students individually without someone else help. 
# Each student presents only their solution. The assignment consists of three parts: 

#     Short description of the problem and method you use to approach it to solve it.
#     Solution implementation. (Python code)
#     Presentation of the solution.

# Requirements for assignments:

#     Students can choose any problem to solve that satisfies requirements and is not solved or not being solved by another student. 
#     Data for the assignment shall be taken from a single dataset from the list provided in the link below.
#     The solution shall be solved by one of the Big data technology approaches/techniques (parallel programming, spark, HPC, or other.)
#     The solution should be different from what we did in other assignments of the course.

# Evaluation of the task: Correctness of the solution, amount of work done, original approach, 
# visualization of data, or any other data or performance related to the solution, 
# creativity (nonstandard solution, approach, problem). The best solution gets the highest grade, other solutions get a relative grade. 

#_________________________________________________________________________________
# 1. Load and preprocess text data

def preprocess(df):
    # remove uneccesary columns
    sub = df.select(["text", "tags"])
    # filter out NA
    sub = sub.filter((df.text.isNotNull() & df.tags.isNotNull()))
    # change data types
    def parse_list(s):
        try:
            return ast.literal_eval(s)
        except Exception:
            return []

    parse_udf = F.udf(parse_list, ArrayType(StringType()))
    sub = sub.withColumn("tags", parse_udf(sub["tags"]))
    return sub


sub = preprocess(df)

# make one tag per row
sub_exploded = sub.select(sub.text, F.explode(sub.tags).alias("tag"))

# Distribution of tags
tag_count = sub_exploded.groupBy("tag").count().orderBy("count")
pdf = tag_count.toPandas()
pdf.to_csv("tag_distribution.csv", index=False)

#_________________________________________________________________________________
# 2. Clean & tokenize text: convert to lower case, remove punctuation, 
# tokenize, remove stopwords

def clean_and_tokenize(df):
    # Converting everything to lower case
    df = df.select(F.lower(df.text).alias("text"), 
                                    F.lower(df.tag).alias("tag"))
    # normalize all apostrophes before stopword removal
    df = df.withColumn("text", F.regexp_replace(F.col("text"),
        "[\u2018\u2019\u201A\u201B\u00B4\u02BB\u02B9\u02C8\u2032]", "'"))
    # Remove all that are not (^): a-z, A-Z or whitespaces
    df = df.withColumn("cleaned_text", F.regexp_replace(F.col("text"), r"[^a-zA-Z\s']", ""))
    # After punctuation removal multiple whitespaces appear
    df = df.withColumn("cleaned_text", F.regexp_replace(F.col("cleaned_text"), r"\s+", " "))
    # tokenize
    tokenizer = ml.Tokenizer(outputCol="tokens", inputCol="cleaned_text")
    df = tokenizer.transform(df)
    # Remove stop words
    stop_words = stopwords.words('english')
    remover = ml.StopWordsRemover(stopWords=stop_words, inputCol="tokens",
                                outputCol="removed_stopwords")
    df = remover.transform(df)
    return df



sub_exploded = clean_and_tokenize(sub_exploded)


#_________________________________________________________________________________
# 3. Generate features (TF, TF-IDF)
# First level - most important according to the word frequency

# use countvectorizer to be able to get the actual words
cv = ml.CountVectorizer(inputCol="removed_stopwords", outputCol="vectors")
model = cv.fit(sub_exploded)
sub_exploded = model.transform(sub_exploded) # first num is vocab size, then indices, then counts

vocab = model.vocabulary

# Convert from sparse vector to array of lenght = vocab len
df_counts = sub_exploded.withColumn("vector_array", vector_to_array("vectors"))

# Aggregate according to tag
df_counts = df_counts.select("tag", F.posexplode("vector_array").alias("word_index", "count"))
df_counts = df_counts.filter(F.col("count") > 0)
vocab_df = spark.createDataFrame(enumerate(vocab), ["word_index", "word"])
df_counts = df_counts.join(vocab_df, on="word_index", how="left")
agg_df = df_counts.groupBy("tag", "word").agg(F.sum("count").alias("total_count"))

# Find the top 3
window = Window.partitionBy("tag").orderBy(F.desc("total_count"))

top_words = agg_df.withColumn("rank", F.row_number().over(window)).filter("rank <= 3")
p_df = top_words.toPandas()
p_df.to_csv("word_count.csv", index=False)

# More options for visualization:
top_words = agg_df.withColumn("rank", F.row_number().over(window)).filter("rank <= 10")
p_df = top_words.toPandas()
p_df.to_csv("word_count_top10.csv", index=False)

top_words = agg_df.withColumn("rank", F.row_number().over(window))
p_df = top_words.toPandas()
p_df.to_csv("word_count_all.csv", index=False)



# Second level - most important according to TF-IDF score
# But this one is not aggregated, meaning that it's checking most important words per individual doc, not the group of docs
# (meaning not group of docs with this tag)
idf = ml.IDF(inputCol="vectors", outputCol="tfidf_vectors")
idf_model = idf.fit(sub_exploded)
sub_exploded = idf_model.transform(sub_exploded)

df_tfidf = sub_exploded.withColumn("tfidf_array", vector_to_array("tfidf_vectors"))
df_tfidf = df_tfidf.select("tag", F.posexplode("tfidf_array").alias("word_index", "tfidf"))
df_tfidf = df_tfidf.filter(F.col("tfidf") > 0)

# df_tfidf = df_tfidf.join(vocab_df, on="word_index", how="left")

# # Find the top 3
# window = Window.partitionBy("tag").orderBy(F.desc("tfidf"))
# top_words_tfidf = df_tfidf.withColumn("rank", F.row_number().over(window)).filter("rank <= 3")
# top_words_tfidf.show()

# Try to aggregate text per tag
# TF-IDF boosts words that are frequent in a document but rare across the corpus. 
# summing accross all docs give the highest scores to those words that were frequent in ALL texts per this tag
agg_df_tfidf = df_tfidf.groupBy("tag", "word_index").agg(F.avg("tfidf").alias("total_tfidf")) # here aggregation is done only if the words appears (0 filtered earlier)
agg_df_tfidf = agg_df_tfidf.join(vocab_df, on="word_index", how="left")
window = Window.partitionBy("tag").orderBy(F.desc("total_tfidf"))
top_words_tfidf_agg = agg_df_tfidf.withColumn("rank", F.row_number().over(window)).filter("rank <= 3")
p_df = top_words_tfidf_agg.toPandas()
p_df.to_csv("word_tfidf_scores.csv", index=False)

# More options for visualization:
top_words_tfidf_agg = agg_df_tfidf.withColumn("rank", F.row_number().over(window)).filter("rank <= 10")
p_df = top_words_tfidf_agg.toPandas()
p_df.to_csv("word_tfidf_scores_top10.csv", index=False)

top_words_tfidf_agg = agg_df_tfidf.withColumn("rank", F.row_number().over(window))
p_df = top_words_tfidf_agg.toPandas()
p_df.to_csv("word_tfidf_scores_all.csv", index=False)

spark.stop()

