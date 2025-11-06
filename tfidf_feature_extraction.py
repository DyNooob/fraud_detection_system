from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

MAX_FEATURES = 1000
NGRAM_RANGE = (1, 2)
MIN_DF = 5
TRAIN_RATIO = 0.7
RANDOM_STATE = 42

def load_data(data_path):
    data = pd.read_csv(data_path)
    print(f"加载数据完成，共{len(data)}条样本")
    print(f"诈骗类型分布：\n{data['label'].value_counts()}")
    plt.figure(figsize=(8, 6))
    data['label'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['刷单返利(1)', '虚假投资(2)', '冒充客服(3)'])
    plt.title('诈骗类型分布')
    plt.ylabel('')
    plt.savefig('data_distribution.png')
    plt.close()
    return data

def preprocess_text(data, stopwords_path):
    stopwords = set()
    with open(stopwords_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.add(word)
    def cut_text(text):
        text = str(text).strip() if pd.notna(text) else ""
        words = jieba.lcut(text)
        filtered_words = [word for word in words if word not in stopwords and len(word) >= 2]
        return " ".join(filtered_words)
    data["cut_content"] = data["content"].apply(cut_text)
    all_words = " ".join(data["cut_content"].tolist())
    wordcloud = WordCloud(font_path="C:/Windows/Fonts/simhei.ttf", width=800, height=400).generate(all_words)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("文本词云图")
    plt.savefig('text_wordcloud.png')
    plt.close()
    return data

def extract_tfidf_features(train_texts, test_texts):
    tfidf = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, min_df=MIN_DF)
    X_train = tfidf.fit_transform(train_texts)
    X_test = tfidf.transform(test_texts)
    feature_names = tfidf.get_feature_names_out()
    top_indices = X_train.sum(axis=0).argsort()[0, -20:].tolist()[0]
    top_words = [feature_names[i] for i in top_indices]
    top_weights = [X_train.sum(axis=0)[0, i] for i in top_indices]
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_weights, y=top_words)
    plt.title('TF-IDF TOP20关键词权重')
    plt.savefig('tfidf_top_words.png')
    plt.close()
    return tfidf, X_train, X_test

def main():
    data = load_data("preprocessed_scam_data.csv")
    data = preprocess_text(data, "merged_stopwords.txt")
    train_data = data.sample(frac=TRAIN_RATIO, random_state=RANDOM_STATE)
    test_data = data.drop(train_data.index)
    tfidf, X_train, X_test = extract_tfidf_features(train_data["cut_content"], test_data["cut_content"])
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    joblib.dump((X_train, X_test), "tfidf_features.pkl")
    joblib.dump((train_data["label"].values, test_data["label"].values), "labels.pkl")
    joblib.dump((train_data["cut_content"].tolist(), test_data["cut_content"].tolist()), "cut_texts.pkl")
    print("TF-IDF特征提取完成，已保存相关文件")

if __name__ == "__main__":
    main()