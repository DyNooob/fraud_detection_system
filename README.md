# fraud_detection_system

## 项目概述
本系统基于机器学习与自然语言处理技术，针对网络诈骗聊天记录进行语义分析、诈骗类型分类及关键证据自动提取。

---

## 系统特性
- 🔍 多模型融合：TF-IDF + Word2Vec + 朴素贝叶斯 + Mini-BERT  
- 🎯 四类识别：正常对话 / 刷单返利 / 虚假投资 / 冒充客服  
- 🔑 证据提取：基于注意力机制的多策略融合定位  
- 📊 可视化分析：热力图、词云、统计图  
- ⚡ 高效处理：单条文本最快仅需 0.006 秒！

---

## 实验性能
- 大类分类（诈骗、正常）准确率：98.1%
- 证据提取精确率：89.17%
- 效率提升：较人工快约 57 倍

---

## 文件结构
```
fraud_detection_system/
├── 核心代码/
│   ├── tfidf_feature_extraction.py
│   ├── word2vec_feature_extraction.py
│   ├── naive_bayes_train.py
│   ├── mini_bert_train.py
│   ├── key_evidence_extraction.py
│   └── efficiency_test.py
├── 数据文件/
│   ├── data.csv
│   ├── preprocessed_scam_data.csv
│   ├── label0-new1030正常.csv
│   └── merged_stopwords.txt
├── 模型文件/
│   ├── mini_bert_scam_model/
│   │   ├── best_model/
│   │   └── best_tokenizer/
│   ├── naive_bayes_tfidf_model.pkl
│   └── tfidf_vectorizer.pkl
├── 实验结果/
│   ├── efficiency_test_result/
│   └── evidence_visualizations/
├── 可视化图表/
│   ├── data_distribution.png
│   ├── confusion_matrix.png
│   ├── confidence_analysis.png
│   └── ...
其中 mini_bert_scam_model 因Github上传限制，已传至百度网盘
```
mini_bert_scam_model 文件夹链接：https://pan.baidu.com/s/1bkYAPsxo7XLs4lzLSo0Pbw?pwd=tli1 
---

## 快速开始

### 环境依赖
```bash
Python >= 3.8
torch >= 1.9.0
transformers >= 4.20.0
scikit-learn >= 1.0.0
jieba >= 0.42.0
gensim >= 4.0.0
matplotlib >= 3.5.0
```

### 运行步骤
```bash
# 特征提取
python tfidf_feature_extraction.py
python word2vec_feature_extraction.py

# 模型训练
python naive_bayes_train.py
python mini_bert_train.py

# 证据提取
python key_evidence_extraction.py

# 效率测试
python efficiency_test.py
```

---

## 核心功能

### 1️⃣ 多模型分类
- 朴素贝叶斯：轻量快速（准确率 86.2%）
- Mini-BERT：高精度语义识别（准确率 92.1%）

### 2️⃣ 关键证据提取
融合三类策略：
- 注意力权重分析
- 敏感词库匹配
- 语义统计特征

### 3️⃣ 可视化分析
- 热力图、词云
- 混淆矩阵
- 置信度与证据权重分布

---


## 引用
> 《融合动态权重与多策略证据提取的 Mini-BERT 诈骗文本检测模型》发表中...

---

## 联系方式
如需技术支持，请联系项目维护者@DyNooob

---
