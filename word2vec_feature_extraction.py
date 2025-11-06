from gensim.models import Word2Vec
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 3
WORKERS = 4
EPOCHS = 10

def load_cut_texts():
    train_texts, test_texts = joblib.load("cut_texts.pkl")
    train_corpus = [text.split() for text in train_texts]
    test_corpus = [text.split() for text in test_texts]
    print(f"加载分词数据完成，训练集{len(train_corpus)}条，测试集{len(test_corpus)}条")
    return train_corpus, test_corpus

def train_word2vec(corpus):
    model = Word2Vec(sentences=corpus, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT, workers=WORKERS, epochs=EPOCHS)
    print(f"Word2Vec模型训练完成，词汇表大小：{len(model.wv)}")
    return model

def get_sentence_vectors(corpus, model):
    def vectorize(sentence):
        vectors = [model.wv[word] for word in sentence if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.zeros(VECTOR_SIZE)
    return np.array([vectorize(sent) for sent in corpus])


def visualize_word_vectors(model):

    all_words = sorted(model.wv.index_to_key, key=lambda x: model.wv.get_vecattr(x, "count"), reverse=True)
    top_words = all_words[:100]



    valid_words = [word for word in top_words if len(word) >= 2]
    if len(valid_words) < 5:
        print("有效词汇不足，跳过可视化")
        return


    word_vectors = np.array([model.wv[word] for word in valid_words])


    perplexity = min(20, len(valid_words) // 2)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
    vectors_2d = tsne.fit_transform(word_vectors)


    plt.figure(figsize=(16, 14))
    for i, word in enumerate(valid_words):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1], s=50, alpha=0.7)
        plt.annotate(
            word,
            (vectors_2d[i, 0], vectors_2d[i, 1]),
            fontsize=10,
            alpha=0.8,
            xytext=(5, 2),
            textcoords='offset points'
        )

    plt.title('Word2Vec高频词向量t-SNE可视化（100个）', fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('word2vec_visualization.png', dpi=300)
    plt.close()

def main():
    train_corpus, test_corpus = load_cut_texts()
    w2v_model = train_word2vec(train_corpus)
    X_train = get_sentence_vectors(train_corpus, w2v_model)
    X_test = get_sentence_vectors(test_corpus, w2v_model)
    visualize_word_vectors(w2v_model)
    w2v_model.save("word2vec_scam_model.model")
    joblib.dump((X_train, X_test), "word2vec_features.pkl")
    print("Word2Vec特征提取完成，已保存相关文件")

if __name__ == "__main__":
    main()