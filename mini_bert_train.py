import torch
import numpy as np
import pandas as pd
import os
import re
import jieba
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from collections import defaultdict
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False


# 1. 全局工具函数：强制清洗所有非字符串数据
def force_clean_text(input_data):
    if pd.isna(input_data) or input_data is None:
        return "无有效内容"
    elif isinstance(input_data, bool):
        return "布尔值数据"
    elif isinstance(input_data, (int, float)):
        return f"数字数据{input_data}"
    elif isinstance(input_data, (list, dict, tuple, set)):
        return str(input_data)[:100]
    elif isinstance(input_data, str):
        clean_str = re.sub(r"\s+", " ", input_data.strip())
        return clean_str if clean_str else "无有效内容"
    else:
        return str(input_data)[:100]


# 2. 设备初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备：{device}")
if device.type == "cuda":
    print(f"显存总量：{torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    print(f"初始剩余显存：{torch.cuda.mem_get_info()[0] / 1024 ** 3:.2f} GB")


# 3. 数据集类（确保标签是列表）
class ScamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128, is_train=False):
        self.texts = [force_clean_text(text) for text in texts]
        self.labels = [int(label) for label in list(labels)]  # 强制转列表+int
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.is_train and text != "无有效内容":
            text = self.augment_text(text)

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=False,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def augment_text(self, text):
        """增强策略：空格+同义词+随机插入特征词"""
        if random.random() < 0.3:
            text = re.sub(r"([，。, .；;：:])", r"\1 ", text)
        synonyms = {
            "刷单": ["刷销量", "刷好评"],
            "垫付": ["预付", "代付"],
            "返现": ["返利", "返钱"],
            "投资": ["理财", "投钱"],
            "客服": ["售后", "客服人员"],
            "验证码": ["校验码", "验证信息"]
        }
        for k, v_list in synonyms.items():
            if k in text and random.random() < 0.25:
                text = text.replace(k, random.choice(v_list))
        if "刷单" in text or "垫付" in text:
            insert_words = ["佣金", "任务单", "提现"]
        elif "投资" in text or "理财" in text:
            insert_words = ["高收益", "开户", "操盘"]
        elif "客服" in text or "验证码" in text:
            insert_words = ["退款", "账户", "冻结"]
        else:
            insert_words = []
        if insert_words and random.random() < 0.2:
            insert_pos = random.randint(0, len(text))
            text = text[:insert_pos] + random.choice(insert_words) + text[insert_pos:]
        return text


# 4. 数据加载（增强版：合并多个数据源）
def load_data(scam_csv_path, normal_csv_path, data_csv_path=None, normal_sample_num=500):
    all_texts = []
    all_labels = []

    # 加载诈骗数据
    if not os.path.exists(scam_csv_path):
        raise FileNotFoundError(f"诈骗数据文件不存在: {scam_csv_path}")
    scam_df = pd.read_csv(scam_csv_path)
    required_cols = ['content', 'label']
    for col in required_cols:
        if col not in scam_df.columns:
            raise ValueError(f"诈骗数据缺少列'{col}'，可用列: {scam_df.columns.tolist()}")

    scam_df['content'] = scam_df['content'].apply(force_clean_text)
    scam_df['label'] = pd.to_numeric(scam_df['label'], errors='coerce')
    scam_df = scam_df[scam_df['content'] != "无有效内容"]
    scam_df = scam_df.dropna(subset=['label'])
    scam_df = scam_df[scam_df['label'].isin([1, 2, 3])]

    # 核心优化：给每类诈骗样本添加专属特征词，帮助模型学习类别差异
    def add_exclusive_feature(text, label):
        feature_map = {
            1: " [刷单返利特征：任务返利]",
            2: " [虚假投资特征：贷款理财]",
            3: " [冒充客服特征：退款快递]"
        }
        return text + feature_map.get(label, "")

    scam_df['content'] = scam_df.apply(
        lambda row: add_exclusive_feature(row['content'], row['label']), axis=1
    )

    scam_texts = scam_df['content'].tolist()
    scam_labels = scam_df['label'].tolist()
    all_texts.extend(scam_texts)
    all_labels.extend(scam_labels)
    print(f"加载诈骗数据：共{len(scam_texts)}条（已添加专属特征词）")

    # 加载正常数据
    if not os.path.exists(normal_csv_path):
        raise FileNotFoundError(f"正常数据文件不存在: {normal_csv_path}")
    normal_df = pd.read_csv(normal_csv_path)
    if 'content' not in normal_df.columns:
        raise ValueError(f"正常数据缺少'content'列，可用列: {normal_df.columns.tolist()}")

    normal_df['content'] = normal_df['content'].apply(force_clean_text)
    normal_df = normal_df[normal_df['content'] != "无有效内容"]

    sample_num = min(normal_sample_num, len(normal_df))
    normal_df_sampled = normal_df.sample(n=sample_num, random_state=42)
    normal_texts = normal_df_sampled['content'].tolist()
    normal_labels = [0] * len(normal_texts)
    all_texts.extend(normal_texts)
    all_labels.extend(normal_labels)
    print(f"加载正常数据：共{len(normal_texts)}条（清洗后，抽样自{len(normal_df)}条）")

    # 加载额外的data.csv数据（如果存在）
    if data_csv_path and os.path.exists(data_csv_path):
        print(f"\n加载额外数据: {data_csv_path}")
        data_df = pd.read_csv(data_csv_path)
        if 'content' in data_df.columns and 'label' in data_df.columns:
            data_df['content'] = data_df['content'].apply(force_clean_text)
            data_df['label'] = pd.to_numeric(data_df['label'], errors='coerce')
            data_df = data_df[data_df['content'] != "无有效内容"]
            data_df = data_df.dropna(subset=['label'])
            data_df = data_df[data_df['label'].isin([0, 1, 2, 3])]

            # 给诈骗数据添加特征词
            def process_data_row(row):
                text = row['content']
                label = row['label']
                if label in [1, 2, 3]:
                    text = add_exclusive_feature(text, label)
                return text

            data_df['content'] = data_df.apply(process_data_row, axis=1)

            data_texts = data_df['content'].tolist()
            data_labels = data_df['label'].tolist()
            all_texts.extend(data_texts)
            all_labels.extend(data_labels)
            print(f"加载额外数据：共{len(data_texts)}条")
        else:
            print("警告: data.csv缺少content或label列")

    total_label_map = {0: "正常对话", 1: "刷单返利", 2: "虚假投资", 3: "冒充客服"}

    # 数据分布可视化
    print(f"\n总样本{len(all_texts)}条，类别分布：")
    label_counts = {}
    for label in [0, 1, 2, 3]:
        count = all_labels.count(label)
        label_counts[total_label_map[label]] = count
        print(f"  - {total_label_map[label]}：{count}条")

    # 绘制数据分布饼图
    plt.figure(figsize=(10, 6))
    plt.pie(label_counts.values(), labels=label_counts.keys(), autopct='%1.1f%%', startangle=90)
    plt.title('数据集类别分布')
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 绘制类别数量柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(label_counts.keys(), label_counts.values(), color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('各类别样本数量')
    plt.xlabel('类别')
    plt.ylabel('样本数量')
    for bar, count in zip(bars, label_counts.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'{count}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        all_texts, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    print(f"数据分割：训练集{len(train_texts)}条，验证集{len(val_texts)}条，测试集{len(test_texts)}条")

    return (train_texts, val_texts, test_texts,
            train_labels, val_labels, test_labels)


# 5. 带动态权重的训练器（改进版）
class DynamicWeightTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 根据类别不平衡调整初始权重
        self.class_weights = torch.tensor([
            1.0,  # 0-正常对话
            2.5,  # 1-刷单返利
            3.0,  # 2-虚假投资
            2.0  # 3-冒充客服
        ], device=device, dtype=torch.float32)

        # 训练历史记录
        self.train_history = {
            'steps': [],
            'train_loss': [],
            'eval_accuracy': [],
            'eval_f1': []
        }

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.class_weights.float())
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        """记录训练损失"""
        loss = super().training_step(model, inputs)
        if self.state.global_step % 50 == 0:
            self.train_history['steps'].append(self.state.global_step)
            self.train_history['train_loss'].append(loss.item())
        return loss

    def evaluate(self, *args, **kwargs):
        eval_result = super().evaluate(*args, **kwargs)

        # 记录评估指标
        if self.state.global_step % 50 == 0:
            self.train_history['eval_accuracy'].append(eval_result.get('eval_accuracy', 0))
            self.train_history['eval_f1'].append(eval_result.get('eval_f1', 0))

        # 动态调整权重逻辑保持不变
        pred_output = self.predict(self.eval_dataset)
        preds = pred_output.predictions.argmax(-1)

        if isinstance(pred_output.label_ids, bool):
            labels = np.array([pred_output.label_ids])
        elif not isinstance(pred_output.label_ids, np.ndarray):
            labels = np.array(pred_output.label_ids)
        else:
            labels = pred_output.label_ids

        error_rates = []
        for label in [0, 1, 2, 3]:
            label_mask = (labels == label)
            total = sum(label_mask)
            if total == 0:
                error_rates.append(0.0)
                continue
            error_count = sum(preds[label_mask] != label)
            error_rates.append(error_count / total)

        self.class_weights = torch.tensor([
            1.0 + error_rates[0],
            2.5 + error_rates[1],
            3.0 + error_rates[2],
            2.0 + error_rates[3]
        ], device=device, dtype=torch.float32)
        print(f"\n动态调整类别权重：{self.class_weights.tolist()}")
        return eval_result

    def plot_training_history(self):
        """绘制训练历史"""
        if not self.train_history['steps']:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # 训练损失
        ax1.plot(self.train_history['steps'], self.train_history['train_loss'])
        ax1.set_title('训练损失变化')
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('损失')
        ax1.grid(True)

        # 评估指标
        if self.train_history['eval_accuracy']:
            eval_steps = self.train_history['steps'][:len(self.train_history['eval_accuracy'])]
            ax2.plot(eval_steps, self.train_history['eval_accuracy'], label='准确率', marker='o')
            ax2.plot(eval_steps, self.train_history['eval_f1'], label='F1分数', marker='s')
            ax2.set_title('验证集性能')
            ax2.set_xlabel('训练步数')
            ax2.set_ylabel('分数')
            ax2.legend()
            ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


# 6. 模型训练
def train_model(tokenizer_path, model_path, scam_csv, normal_csv, data_csv=None):
    # 使用本地BERT模型
    local_bert_path = r"D:\Projects\Python\Fraud_Messages_Detection\bert-base-chinese"

    if os.path.exists(local_bert_path):
        print(f"使用本地BERT模型: {local_bert_path}")
        tokenizer = BertTokenizerFast.from_pretrained(local_bert_path)
    else:
        print("警告: 未找到本地BERT模型，尝试从网络加载...")
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    tokenizer.save_pretrained(tokenizer_path)
    print(f"Tokenizer已保存到: {tokenizer_path}")

    # 加载数据（包含额外数据）
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_data(
        scam_csv, normal_csv, data_csv
    )

    # 创建数据集
    class ScamDatasetWithMeta(ScamDataset):
        def __init__(self, texts, labels, tokenizer, max_len=128, is_train=False):
            if is_train:
                processed_texts = texts
            else:
                processed_texts = [
                    text.replace(" [刷单返利特征：任务返利]", "").replace(" [虚假投资特征：贷款理财]", "").replace(
                        " [冒充客服特征：退款快递]", "") for text in texts]
            super().__init__(processed_texts, labels, tokenizer, max_len, is_train)
            self.raw_texts = processed_texts
            self.raw_labels = labels

    train_dataset = ScamDatasetWithMeta(train_texts, train_labels, tokenizer, is_train=True)
    val_dataset = ScamDatasetWithMeta(val_texts, val_labels, tokenizer)
    test_dataset = ScamDatasetWithMeta(test_texts, test_labels, tokenizer)

    # 加载模型
    if os.path.exists(local_bert_path):
        print(f"从本地加载BERT模型: {local_bert_path}")
        model = BertForSequenceClassification.from_pretrained(
            local_bert_path,
            num_labels=4,
            ignore_mismatched_sizes=True
        )
    else:
        print("从预训练模型初始化...")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-chinese",
            num_labels=4
        )

    model.to(device)
    print("模型加载成功")

    # 评估指标
    def compute_metrics(pred):
        preds = pred.predictions.argmax(-1)
        if isinstance(pred.label_ids, bool):
            labels = np.array([pred.label_ids])
        else:
            labels = pred.label_ids
        report = classification_report(
            labels, preds, target_names=["正常对话", "刷单返利", "虚假投资", "冒充客服"],
            output_dict=True, zero_division=0
        )
        return {
            'accuracy': report['accuracy'],
            'f1': report['macro avg']['f1-score'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall']
        }

    # 训练参数（根据数据量调整）
    total_steps = len(train_dataset) * 12 // 8  # 估计总步数
    training_args = TrainingArguments(
        output_dir='./training_results',
        num_train_epochs=12,  # 增加训练轮次
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=min(500, total_steps // 10),
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy='steps',
        eval_steps=min(200, total_steps // 20),
        save_strategy='steps',
        save_steps=min(200, total_steps // 20),
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        learning_rate=2e-5,
        lr_scheduler_type='linear',
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        report_to='none'
    )

    trainer = DynamicWeightTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\n开始训练...")
    trainer.train()
    trainer.plot_training_history()

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"模型保存至 {model_path}")

    return tokenizer, model, test_dataset, trainer


# 7. 证据提取和标签映射
type_exclusive_features = {
    "刷单返利": ["返利", "佣金", "点赞赚钱", "做任务", "垫付", "刷销量", "任务返利"],
    "虚假投资": ["贷款", "借款", "利率", "额度", "投资", "理财", "高收益", "贷款理财"],
    "冒充客服": ["快递问题", "退款", "账户异常", "商品召回", "注销白条", "退款快递"],
    "正常对话": ["发货", "物流", "价格", "好评", "收货", "优惠"]
}

label_to_scam_type = {0: "正常对话", 1: "刷单返利", 2: "虚假投资", 3: "冒充客服"}


def extract_key_evidence(text, tokenizer, model):
    text_clean = force_clean_text(text)
    text_clean = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text_clean).strip()

    # 改进：更智能的特征词匹配
    type_scores = {}
    for typ, words in type_exclusive_features.items():
        score = sum(1 for word in words if word in text_clean)
        type_scores[typ] = score

    # 只有当某个类别的特征词明显多于其他类别时才强制匹配
    forced_type = None
    if type_scores:
        max_score = max(type_scores.values())
        if max_score >= 2:  # 至少匹配2个特征词才强制
            for typ, score in type_scores.items():
                if score == max_score:
                    forced_type = typ
                    break

    # 模型推理
    inputs = tokenizer(
        text_clean, return_tensors="pt", truncation=True, padding="max_length", max_length=128
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_label = np.argmax(probs)
    pred_type = label_to_scam_type[pred_label]
    confidence = round(probs[pred_label] * 100, 2)

    if forced_type and forced_type != pred_type:
        print(f"  特征词强制匹配: {pred_type} -> {forced_type}")
        pred_type = forced_type
        confidence = min(99.0, confidence + 10)  # 适当提高置信度

    # 证据提取
    evidence = []
    for word in type_exclusive_features[pred_type]:
        if word in text_clean and len(evidence) < 3:
            evidence.append(word)

    if len(evidence) < 3:
        cut_words = jieba.lcut(text_clean)
        related_words = [
            w for w in cut_words
            if any(feat in w for feat in type_exclusive_features[pred_type])
               and 2 <= len(w) <= 5
               and w not in ["无有效内容", "布尔值数据", "数字数据"]
        ]
        evidence += [w for w in related_words if w not in evidence][:3 - len(evidence)]

    if len(evidence) < 3:
        cut_words = jieba.lcut(text_clean)
        valid_words = [w for w in cut_words if 2 <= len(w) <= 5 and w not in ["的", "了", "在"]]
        evidence += [w for w in valid_words if w not in evidence][:3 - len(evidence)]

    return evidence, pred_type, confidence, pred_label


# 8. 用CSV测试集验证（增强版，包含可视化）
def validate_with_csv_data(tokenizer, model, test_dataset, sample_num=50):
    # 从测试集中随机抽样
    sample_indices = random.sample(range(len(test_dataset)), min(sample_num, len(test_dataset)))
    texts = [test_dataset.raw_texts[i] for i in sample_indices]
    true_labels = [test_dataset.raw_labels[i] for i in sample_indices]
    true_types = [label_to_scam_type[label] for label in true_labels]

    print("\n" + "=" * 80)
    print(f"用CSV测试集验证（随机抽样{len(texts)}条）")
    print("=" * 80)

    all_preds = []
    all_true = []
    all_confidences = []
    all_evidences = []

    correct = 0
    for i, (text, true_label, true_type) in enumerate(zip(texts, true_labels, true_types), 1):
        evidence, pred_type, confidence, pred_label = extract_key_evidence(text, tokenizer, model)

        all_preds.append(pred_label)
        all_true.append(true_label)
        all_confidences.append(confidence)
        all_evidences.append(evidence)

        # 截断长文本显示
        display_text = text[:120] + "..." if len(text) > 120 else text
        print(f"\n【测试样本{i}】")
        print(f"文本：{display_text}")
        print(f"真实类型：{true_type}")
        print(f"预测类型：{pred_type}（可信度：{confidence}%）")
        print(f"关键证据：{evidence}")
        if pred_type == true_type:
            correct += 1
            print("✅  预测正确")
        else:
            print("❌  预测错误")

    # 计算准确率
    acc = round(correct / len(texts) * 100, 2)
    print(f"\n验证总结：共{len(texts)}条，正确率{acc}%")
    print("=" * 80)

    # 生成详细性能报告
    generate_performance_report(all_true, all_preds, all_confidences, texts, all_evidences)

    return acc


def generate_performance_report(true_labels, pred_labels, confidences, texts, evidences):
    """生成详细的性能报告和可视化"""

    # 1. 分类报告
    print("\n【详细分类报告】")
    report = classification_report(
        true_labels, pred_labels,
        target_names=[label_to_scam_type[i] for i in range(4)],
        digits=4
    )
    print(report)

    # 保存报告到文件
    with open('classification_report.txt', 'w', encoding='utf-8') as f:
        f.write("Mini-BERT 诈骗检测模型分类报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(report)

    # 2. 混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label_to_scam_type[i] for i in range(4)],
                yticklabels=[label_to_scam_type[i] for i in range(4)])
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 各类别准确率
    class_accuracies = {}
    for label in range(4):
        mask = np.array(true_labels) == label
        if sum(mask) > 0:
            accuracy = (np.array(pred_labels)[mask] == label).mean()
            class_accuracies[label_to_scam_type[label]] = accuracy * 100

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_accuracies.keys(), class_accuracies.values(),
                   color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.title('各类别准确率')
    plt.ylabel('准确率 (%)')
    plt.ylim(0, 100)

    # 在柱子上添加数值
    for bar, acc in zip(bars, class_accuracies.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 置信度分布
    plt.figure(figsize=(12, 6))

    # 正确和错误预测的置信度分布
    correct_mask = np.array(pred_labels) == np.array(true_labels)
    correct_conf = np.array(confidences)[correct_mask]
    wrong_conf = np.array(confidences)[~correct_mask]

    plt.subplot(1, 2, 1)
    if len(correct_conf) > 0:
        plt.hist(correct_conf, bins=10, alpha=0.7, label='正确预测', color='green')
    if len(wrong_conf) > 0:
        plt.hist(wrong_conf, bins=10, alpha=0.7, label='错误预测', color='red')
    plt.xlabel('置信度 (%)')
    plt.ylabel('频次')
    plt.title('预测置信度分布')
    plt.legend()

    # 各类别平均置信度
    plt.subplot(1, 2, 2)
    avg_conf_by_class = []
    for label in range(4):
        mask = np.array(pred_labels) == label
        if sum(mask) > 0:
            avg_conf = np.mean(np.array(confidences)[mask])
        else:
            avg_conf = 0
        avg_conf_by_class.append(avg_conf)

    plt.bar([label_to_scam_type[i] for i in range(4)], avg_conf_by_class, color='orange')
    plt.xlabel('预测类别')
    plt.ylabel('平均置信度 (%)')
    plt.title('各类别平均预测置信度')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 可视化结果已保存:")
    print(f"   - confusion_matrix.png (混淆矩阵)")
    print(f"   - class_accuracy.png (各类别准确率)")
    print(f"   - confidence_analysis.png (置信度分析)")
    print(f"   - classification_report.txt (详细分类报告)")


# 主函数
if __name__ == "__main__":
    # 修复路径配置 - 使用正确的路径
    base_dir = r"D:\Projects\Python\Fraud_Messages_Detection\final_code"
    model_path = os.path.join(base_dir, "mini_bert_scam_model", "best_model")
    tokenizer_path = os.path.join(base_dir, "mini_bert_scam_model", "best_tokenizer")

    scam_csv = "preprocessed_scam_data.csv"  # 诈骗数据CSV
    normal_csv = "label0-new1030正常.csv"  # 正常数据CSV
    data_csv = "data.csv"  # 新增数据源

    # 创建保存目录
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(tokenizer_path, exist_ok=True)

    print(f"模型保存路径: {model_path}")
    print(f"Tokenizer保存路径: {tokenizer_path}")

    # 训练模型
    try:
        tokenizer, model, test_dataset, trainer = train_model(
            tokenizer_path, model_path, scam_csv, normal_csv, data_csv
        )
    except Exception as e:
        print(f"训练失败：{e}")
        import traceback

        traceback.print_exc()
        exit(1)

    # 验证（用CSV测试集随机抽样）
    accuracy = validate_with_csv_data(tokenizer, model, test_dataset, sample_num=50)

    print(f"\n🎉 训练和验证完成！最终测试准确率: {accuracy}%")
    print(f"📁 模型保存在: {model_path}")
    print(f"📊 可视化图表已生成在当前目录")