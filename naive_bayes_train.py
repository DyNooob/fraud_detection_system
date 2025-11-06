import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

PARAM_GRID = {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}  # 扩展参数范围
CV = 5

def load_data():
    try:
        X_train, X_test = joblib.load("tfidf_features.pkl")
        y_train, y_test = joblib.load("labels.pkl")
        print(f"加载数据完成，训练集{X_train.shape}，测试集{X_test.shape}")
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        print(f"错误：未找到数据文件！{str(e)}")
        exit()

def train_model(X_train, y_train):
    nb_model = MultinomialNB()
    # 网格搜索寻找最优参数
    grid_search = GridSearchCV(
        estimator=nb_model,
        param_grid=PARAM_GRID,
        cv=CV,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    # 交叉验证结果可视化
    results = pd.DataFrame(grid_search.cv_results_)
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        x=results["param_alpha"],
        y=results["mean_test_score"],
        marker="o"
    )
    plt.xlabel("平滑参数 alpha")
    plt.ylabel("交叉验证F1分数")
    plt.title("不同alpha值对应的模型性能")
    plt.savefig('nb_alpha_tuning.png')
    plt.close()

    print(f"最优参数：{grid_search.best_params_}，交叉验证F1：{grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # 训练集与测试集性能对比
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_f1 = f1_score(y_train, y_pred_train, average="macro")
    test_f1 = f1_score(y_test, y_pred_test, average="macro")

    # 绘制训练/测试性能对比图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(x=["训练集", "测试集"], y=[train_acc, test_acc])
    plt.ylim(0.7, 1.0)
    plt.title("准确率对比")

    plt.subplot(1, 2, 2)
    sns.barplot(x=["训练集", "测试集"], y=[train_f1, test_f1])
    plt.ylim(0.7, 1.0)
    plt.title("Macro F1对比")
    plt.tight_layout()
    plt.savefig('nb_train_test_comparison.png')
    plt.close()

    # 详细分类报告
    print("\n测试集分类报告：")
    print(classification_report(
        y_test, y_pred_test,
        target_names=["刷单返利", "虚假投资", "冒充客服"],
        digits=4
    ))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["刷单返利", "虚假投资", "冒充客服"],
        yticklabels=["刷单返利", "虚假投资", "冒充客服"]
    )
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("朴素贝叶斯混淆矩阵")
    plt.savefig('nb_confusion_matrix.png')
    plt.close()

    # 各类型F1分数
    metrics = classification_report(
        y_test, y_pred_test,
        target_names=["刷单返利", "虚假投资", "冒充客服"],
        output_dict=True
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=["刷单返利", "虚假投资", "冒充客服", "平均"],
        y=[
            metrics["刷单返利"]["f1-score"],
            metrics["虚假投资"]["f1-score"],
            metrics["冒充客服"]["f1-score"],
            metrics["macro avg"]["f1-score"]
        ]
    )
    plt.ylim(0, 1.0)
    plt.title("各类型F1分数")
    plt.savefig('nb_f1_scores.png')
    plt.close()

    return test_acc, test_f1

def main():
    X_train, X_test, y_train, y_test = load_data()
    best_model = train_model(X_train, y_train)
    acc, f1 = evaluate_model(best_model, X_train, X_test, y_train, y_test)
    print(f"测试集准确率：{acc:.4f}，F1值：{f1:.4f}")
    joblib.dump(best_model, "naive_bayes_tfidf_model.pkl")
    print("朴素贝叶斯模型保存完成")

if __name__ == "__main__":
    main()