import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

tag_dist = pd.read_csv("tag_distribution.csv")
count_df3 = pd.read_csv("word_count.csv")
count_df10 = pd.read_csv("word_count_top10.csv")
tfidf_df3 = pd.read_csv("word_tfidf_scores.csv")
tfidf_df10 = pd.read_csv("word_tfidf_scores_top10.csv")
counts = pd.read_csv("word_count_all.csv")
tfidfs = pd.read_csv("word_tfidf_scores_all.csv")
#_____________________________________________________________________________________
# Visualization of the tag distribution
# Take only the frequent ones, let's say 10 and above
tag_dist = tag_dist.loc[tag_dist["count"] > 30]
plt.bar(tag_dist["tag"], tag_dist["count"])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

five_most_frequent_tags = tag_dist.tag.tail(5).to_list()
five_most_frequent_tags = [x.lower() for x in five_most_frequent_tags]

ten_most_frequent_tags = tag_dist.tag.tail(10).to_list()
ten_most_frequent_tags = [x.lower() for x in ten_most_frequent_tags]

#_____________________________________________________________________________________
# Barplots
# The most frequent tag with it's most freqent words

# Counts
count_df_five = count_df3.loc[count_df3["tag"].isin(five_most_frequent_tags)]
plt.figure(figsize=(12, 6))
sns.barplot(data=count_df_five, y="total_count", x="word", hue="tag")
plt.title("Top Frequent Words per Tag")
plt.xlabel("Word")
plt.ylabel("Frequency")
plt.legend(title="Tag")
plt.tight_layout()
plt.show()

# Tfidf scores
tfidf_df_five = tfidf_df3.loc[tfidf_df3["tag"].isin(five_most_frequent_tags)]
plt.figure(figsize=(12, 6))
sns.barplot(data=tfidf_df_five, y="total_tfidf", x="word", hue="tag")
plt.title("Top TFIDF Words per Tag")
plt.xlabel("Word")
plt.ylabel("Tfidf score")
plt.legend(title="Tag")
plt.tight_layout()
plt.show()

#_____________________________________________________________________________________
# Heatmap
# Interaction between tags and words. Helps to see which words are very frequent in other tags too

# Counts
count_df_ten = count_df3.loc[count_df3["tag"].isin(ten_most_frequent_tags)]
pivot = count_df_ten.pivot(index="word", columns="tag", values="total_count").fillna(0)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, cmap="YlGnBu", annot=False)
plt.title("Word frequency Heatmap: Words vs Tags")
plt.xlabel("Tag")
plt.ylabel("Word")
plt.tight_layout()
plt.show()

# Tfidf scores
tfidf_df_ten= tfidf_df3.loc[tfidf_df3["tag"].isin(ten_most_frequent_tags)]
pivot = tfidf_df_ten.pivot(index="word", columns="tag", values="total_tfidf").fillna(0)

plt.figure(figsize=(10, 8))
sns.heatmap(pivot, cmap="YlGnBu", annot=False)
plt.title("Word TFIDF score Heatmap: Words vs Tags")
plt.xlabel("Tag")
plt.ylabel("Word")
plt.tight_layout()
plt.show()

#_____________________________________________________________________________________
# Wordclouds

# Counts
# tag = "writing" # most frequent in this subset
tag = "mental health" 
subset_c = count_df10[count_df10["tag"] == tag]
word_freq = dict(zip(subset_c["word"], subset_c["total_count"]))
cloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

plt.subplot(1,2,1)
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.title(f"Top Words for Tag: {tag}. Word counts")


# Tfidf scores
subset_t = tfidf_df10[tfidf_df10["tag"] == tag]
word_freq = dict(zip(subset_t["word"], subset_t["total_tfidf"]))
cloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

plt.subplot(1,2,2)
plt.imshow(cloud, interpolation="bilinear")
plt.axis("off")
plt.title(f"Top Words for Tag: {tag}. TF-IDF scores")
plt.show()

#_____________________________________________________________________________________
# Comparison
# Print table of words: count vs tfidf and compare the overlapping
print(subset_c)
print(subset_t)
#_____________________________________________________________________________________
#  Scatter plot per tag
# words with high frequency, but low score - common, but not important
# words with low frequency, but high score - rare, but distinctive
# need the same words
tag = "startup" 
subset_c = counts[counts["tag"] == tag]
subset_t = tfidfs[tfidfs["tag"] == tag]

same_words = subset_c.merge(subset_t, on="word", how="inner") # of course, they have same words, because its a vocab with more than 1 word?
plt.scatter(same_words["total_count"], same_words["total_tfidf"])
plt.xlabel("Word Counts")
plt.ylabel("TF-IDF scores")
plt.title(f"Word counts vs TF-IDF scores for tag {tag}")
for _, row in same_words.iterrows():
    if row['total_tfidf'] > 100:  
        plt.text(row['total_count'], row['total_tfidf'], row['word'])
    if row['total_count'] > 600: 
        plt.text(row['total_count'], row['total_tfidf'], row['word'])
plt.show()
