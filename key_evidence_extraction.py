# key_evidence_extraction.py - 关键证据提取与可视化系统（使用外部停用词文件）
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizerFast, BertForSequenceClassification
import re
import jieba
from collections import defaultdict, Counter
import os
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


# 1. 修复中文显示问题
def setup_chinese_font():
    """设置中文字体，解决中文显示问题"""
    try:
        # 方法1: 使用系统自带的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 方法2: 检查并设置可用字体
        import matplotlib.font_manager as fm
        fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong', 'STSong', 'STKaiti']

        available_fonts = []
        for font in chinese_fonts:
            if font in fonts:
                available_fonts.append(font)
                print(f"✅ 找到中文字体: {font}")

        if available_fonts:
            plt.rcParams['font.sans-serif'] = available_fonts + ['DejaVu Sans']
            print(f"✅ 使用中文字体: {available_fonts[0]}")
        else:
            print("⚠️  未找到系统中文字体，尝试使用默认字体")

    except Exception as e:
        print(f"❌ 字体设置失败: {e}")


# 初始化中文字体
setup_chinese_font()
sns.set_style("whitegrid")
sns.set_palette("husl")

# 2. 设备初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用计算设备：{device}")


# 3. 加载停用词函数
def load_stopwords(stopwords_file="merged_stopwords.txt"):
    """从文件加载停用词"""
    stopwords = set()
    try:
        if os.path.exists(stopwords_file):
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:  # 跳过空行
                        stopwords.add(word)
            print(f"✅ 从 {stopwords_file} 加载了 {len(stopwords)} 个停用词")
        else:
            print(f"⚠️  停用词文件 {stopwords_file} 不存在，使用默认停用词")
            # 使用默认停用词作为备选
            stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也',
                         '很',
                         '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
    except Exception as e:
        print(f"❌ 加载停用词失败: {e}")
        stopwords = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很',
                     '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}

    return stopwords


# 全局停用词集合
STOPWORDS = load_stopwords()


# 4. 文本清洗函数
def force_clean_text(input_data):
    """强制清洗所有非字符串数据"""
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


# 5. 模型加载（增强错误处理）
def load_bert_model(model_path, tokenizer_path, num_labels=4):
    """加载BERT模型和tokenizer，支持多种错误处理"""
    try:
        # 检查路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer路径不存在: {tokenizer_path}")

        print("正在加载Tokenizer...")
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

        print("正在加载模型...")
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            output_attentions=True,
            output_hidden_states=True,
            ignore_mismatched_sizes=True
        )

        model.eval()
        model.to(device)
        print("✅ 模型和Tokenizer加载成功！")

        # 打印模型信息
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"分类头维度: {model.config.num_labels}")

        return tokenizer, model

    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        print("请检查：")
        print("1. 模型文件是否存在")
        print("2. 模型文件是否完整")
        print("3. 标签数量是否匹配")
        raise


# 6. 敏感词库和标签映射（扩展版）
scam_sensitive_words = {
    "正常对话": ["你好", "谢谢", "请问", "发货", "物流", "价格", "优惠", "收货", "好评"],
    "刷单返利": ["刷单", "垫付", "返利", "佣金", "任务", "点赞", "关注", "销量", "好评返现", "做任务", "本金", "提现"],
    "虚假投资": ["投资", "理财", "贷款", "收益", "利率", "额度", "开户", "操盘", "高收益", "稳赚", "内部消息",
                 "资金盘"],
    "冒充客服": ["客服", "退款", "验证码", "账户", "冻结", "解冻", "快递", "订单异常", "售后", "注销", "白条", "信用"]
}

label_to_scam_type = {0: "正常对话", 1: "刷单返利", 2: "虚假投资", 3: "冒充客服"}
scam_type_to_label = {v: k for k, v in label_to_scam_type.items()}


# 7. 增强版关键证据提取函数
class EvidenceExtractor:
    def __init__(self, tokenizer, model, device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.attention_history = []

    def extract_key_evidence(self, text, top_k=5, max_length=128, return_analysis=False):
        """
        提取关键证据 - 增强版
        返回：证据词列表，诈骗类型，置信度，详细分析（可选）
        """
        # 文本预处理
        text_clean = force_clean_text(text)
        original_text = text_clean

        # 文本编码
        inputs = self.tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_offsets_mapping=True,
            return_attention_mask=True
        )

        offsets = inputs.pop("offset_mapping").cpu().numpy()[0]
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 模型推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            attentions = outputs.attentions

        # 预测结果
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = np.argmax(probs)
        pred_label = max(0, min(pred_label, len(label_to_scam_type) - 1))
        pred_scam_type = label_to_scam_type[pred_label]
        pred_confidence = round(probs[pred_label] * 100, 2)

        # 多策略证据提取
        evidence_tokens = self._extract_evidence_multiple_strategies(
            text_clean, offsets, attentions, top_k, pred_scam_type
        )

        # 记录注意力信息用于可视化
        self.attention_history.append({
            'text': text_clean,
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'attention': attentions,
            'pred_type': pred_scam_type,
            'confidence': pred_confidence
        })

        if return_analysis:
            analysis = {
                'original_text': original_text,
                'cleaned_text': text_clean,
                'probabilities': self._format_probabilities(probs),
                'top_attention_tokens': evidence_tokens[:top_k],
                'sensitive_words_found': self._find_sensitive_words(text_clean, pred_scam_type),
                'text_length': len(text_clean),
                'token_count': len(self.tokenizer.tokenize(text_clean))
            }
            return evidence_tokens[:top_k], pred_scam_type, pred_confidence, analysis

        return evidence_tokens[:top_k], pred_scam_type, pred_confidence

    def _format_probabilities(self, probs):
        """格式化概率输出"""
        return {label_to_scam_type[i]: round(prob * 100, 2) for i, prob in enumerate(probs)}

    def _extract_evidence_multiple_strategies(self, text, offsets, attentions, top_k, pred_type):
        """多策略证据提取"""
        strategies = []

        # 策略1: 注意力权重分析
        strategies.append(self._attention_based_extraction(text, offsets, attentions, top_k))

        # 策略2: 敏感词匹配
        strategies.append(self._sensitive_word_extraction(text, pred_type, top_k))

        # 策略3: 文本分词分析
        strategies.append(self._text_segmentation_extraction(text, top_k))

        # 合并策略结果（去重，按优先级排序）
        all_evidence = []
        seen_words = set()

        for strategy_evidence in strategies:
            for word, score in strategy_evidence:
                if word not in seen_words and len(word) >= 2:
                    all_evidence.append((word, score))
                    seen_words.add(word)

        # 按分数排序并返回前top_k个
        all_evidence.sort(key=lambda x: x[1], reverse=True)
        return [word for word, score in all_evidence[:top_k * 2]]

    def _attention_based_extraction(self, text, offsets, attentions, top_k):
        """基于注意力权重的证据提取"""
        # 使用最后一层的注意力权重
        last_layer_att = attentions[-1].cpu().numpy()[0]

        # 平均所有注意力头，并取[CLS] token对其他token的注意力
        cls_attention = last_layer_att.mean(axis=0)[0]

        # 获取token列表
        tokens = self.tokenizer.convert_ids_to_tokens(
            self.tokenizer(text, truncation=True, max_length=128)['input_ids']
        )

        # 合并subword tokens
        token_scores = []
        current_token = ""
        current_score = 0.0
        token_count = 0

        for i, (token, score) in enumerate(zip(tokens, cls_attention)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue

            if token.startswith('##'):
                # subword token
                current_token += token[2:]
                current_score += score
                token_count += 1
            else:
                # 新token开始，保存前一个token
                if current_token:
                    token_scores.append((current_token, current_score / max(1, token_count)))
                current_token = token
                current_score = score
                token_count = 1

        # 添加最后一个token
        if current_token:
            token_scores.append((current_token, current_score / max(1, token_count)))

        # 过滤和排序
        token_scores = [(token, score) for token, score in token_scores
                        if len(token) >= 2 and not token.isdigit()]
        token_scores.sort(key=lambda x: x[1], reverse=True)

        return token_scores[:top_k]

    def _sensitive_word_extraction(self, text, pred_type, top_k):
        """基于敏感词库的证据提取"""
        sensitive_words = scam_sensitive_words.get(pred_type, [])
        found_words = []

        for word in sensitive_words:
            if word in text:
                # 根据出现位置和频率给分
                count = text.count(word)
                position = text.find(word)
                score = count * 10 + (1 - position / len(text)) * 5
                found_words.append((word, score))

        found_words.sort(key=lambda x: x[1], reverse=True)
        return found_words[:top_k]

    def _text_segmentation_extraction(self, text, top_k):
        """基于文本分词的证据提取 - 使用外部停用词文件"""
        words = jieba.lcut(text)
        word_freq = Counter(words)

        # 使用从文件加载的停用词
        significant_words = [(word, freq) for word, freq in word_freq.items()
                             if len(word) >= 2 and word not in STOPWORDS and not word.isdigit()]

        significant_words.sort(key=lambda x: x[1], reverse=True)
        return [(word, freq * 2) for word, freq in significant_words[:top_k]]

    def _find_sensitive_words(self, text, pred_type):
        """查找文本中的敏感词"""
        sensitive_words = scam_sensitive_words.get(pred_type, [])
        found = [word for word in sensitive_words if word in text]
        return found


# 8. 修复中文显示的可视化功能
class ChineseVisualizer:
    def __init__(self, output_dir="evidence_visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 设置美观的颜色方案
        self.colors = {
            '刷单返利': '#FF6B6B',
            '虚假投资': '#4ECDC4',
            '冒充客服': '#45B7D1',
            '正常对话': '#96CEB4',
            'background': '#F8F9FA',
            'grid': '#E9ECEF'
        }

        # 确保使用中文字体
        self._setup_plot_font()

    def _setup_plot_font(self):
        """为每个图表单独设置字体"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass

    def plot_attention_heatmap(self, extractor, sample_index=0):
        """绘制注意力热力图 - 单独显示"""
        if not extractor.attention_history:
            print("没有注意力历史数据")
            return

        sample = extractor.attention_history[sample_index]
        attentions = sample['attention']

        # 使用最后一层的平均注意力
        last_layer_att = attentions[-1].cpu().numpy()[0]
        avg_attention = last_layer_att.mean(axis=0)

        # 获取有效token（去除特殊token）
        all_tokens = sample['tokens']
        valid_indices = []
        valid_tokens = []

        for i, token in enumerate(all_tokens):
            if token not in ['[CLS]', '[SEP]', '[PAD]'] and len(token.strip()) > 0:
                valid_indices.append(i)
                # 清理token显示
                clean_token = token.replace('##', '')
                if len(clean_token) > 6:
                    clean_token = clean_token[:6] + '..'
                valid_tokens.append(clean_token)

        # 只取前10个有效token，避免过于拥挤
        display_count = min(10, len(valid_indices))
        display_indices = valid_indices[:display_count]
        display_attention = avg_attention[display_indices, :][:, display_indices]

        # 创建热力图
        plt.figure(figsize=(10, 8))

        # 创建自定义颜色映射
        cmap = plt.cm.YlOrRd
        cmap.set_bad(color='white')

        # 绘制热力图
        im = plt.imshow(display_attention, cmap=cmap, aspect='auto')

        # 设置热力图样式 - 完全移除坐标轴标签
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f'注意力权重热力图\n样本 {sample_index + 1} - {sample["pred_type"]} | 置信度: {sample["confidence"]}%',
            fontsize=14, fontweight='bold', pad=20)

        # 添加颜色条
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('注意力权重', rotation=270, labelpad=15)

        # 在热力图上添加数值（只显示较高的权重）
        for i in range(display_count):
            for j in range(display_count):
                value = display_attention[i, j]
                if value > 0.15:  # 只显示显著权重
                    color = 'white' if value > 0.5 else 'black'
                    plt.text(j, i, f'{value:.2f}', ha='center', va='center',
                             color=color, fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/attention_heatmap_{sample_index}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_token_weights(self, extractor, sample_index=0):
        """绘制证据词权重排名 - 单独显示，修复文字竖排问题"""
        if not extractor.attention_history:
            print("没有注意力历史数据")
            return

        sample = extractor.attention_history[sample_index]
        attentions = sample['attention']

        # 使用最后一层的平均注意力
        last_layer_att = attentions[-1].cpu().numpy()[0]
        avg_attention = last_layer_att.mean(axis=0)

        # 获取有效token（去除特殊token）
        all_tokens = sample['tokens']
        valid_indices = []
        valid_tokens = []

        for i, token in enumerate(all_tokens):
            if token not in ['[CLS]', '[SEP]', '[PAD]'] and len(token.strip()) > 0:
                valid_indices.append(i)
                # 清理token显示
                clean_token = token.replace('##', '')
                if len(clean_token) > 6:
                    clean_token = clean_token[:6] + '..'
                valid_tokens.append(clean_token)

        # 只取前10个有效token
        display_count = min(10, len(valid_indices))
        display_indices = valid_indices[:display_count]
        display_attention = avg_attention[display_indices, :][:, display_indices]

        # 计算token权重
        token_scores = np.mean(display_attention, axis=1)

        # 创建权重排名图
        plt.figure(figsize=(12, 8))

        # 创建水平条形图
        y_pos = np.arange(display_count)
        colors = plt.cm.YlOrRd((token_scores - token_scores.min()) / (token_scores.max() - token_scores.min() + 1e-8))

        bars = plt.barh(y_pos, token_scores, color=colors,
                        edgecolor='white', linewidth=1, height=0.7)

        # 关键修复：确保y轴标签水平显示且不重叠
        plt.yticks(y_pos, valid_tokens[:display_count], fontsize=12, rotation=0)  # rotation=0确保水平

        plt.xlabel('平均注意力权重', fontsize=12)
        plt.title(
            f'关键证据词权重排名\n样本 {sample_index + 1} - {sample["pred_type"]} | 置信度: {sample["confidence"]}%',
            fontsize=14, fontweight='bold', pad=20)

        # 在每个条形上添加数值
        for i, (bar, score) in enumerate(zip(bars, token_scores)):
            plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f'{score:.3f}', ha='left', va='center', fontsize=11, fontweight='bold')

        # 设置x轴范围，为文字留出空间
        x_max = max(token_scores) * 1.15
        plt.xlim(0, x_max)

        plt.gca().invert_yaxis()  # 让权重高的在顶部
        plt.grid(True, axis='x', alpha=0.3, linestyle='--')

        # 美化边框
        for spine in plt.gca().spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color('#DDDDDD')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/token_weights_{sample_index}.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_confidence_analysis(self, results):
        """绘制置信度分析图表"""
        if not results:
            return

        confidences = [result['confidence'] for result in results]
        types = [result['pred_type'] for result in results]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('模型置信度综合分析', fontsize=18, fontweight='bold', y=0.95)

        # 子图1: 各类别置信度分布
        type_data = {}
        type_colors = []
        for scam_type in label_to_scam_type.values():
            type_conf = [conf for conf, typ in zip(confidences, types) if typ == scam_type]
            if type_conf:
                type_data[scam_type] = type_conf
                type_colors.append(self.colors.get(scam_type, '#999999'))

        if type_data:
            boxes = ax1.boxplot(type_data.values(), labels=type_data.keys(), patch_artist=True)
            for patch, color in zip(boxes['boxes'], type_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax1.set_title('各类别置信度分布', fontsize=14, fontweight='bold', pad=15)
            ax1.set_ylabel('置信度 (%)', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

        # 子图2: 置信度分布直方图
        ax2.hist(confidences, bins=15, alpha=0.7, color='#4ECDC4', edgecolor='white', linewidth=1.5)
        ax2.set_xlabel('置信度 (%)', fontsize=12)
        ax2.set_ylabel('频次', fontsize=12)
        ax2.set_title('置信度分布直方图', fontsize=14, fontweight='bold', pad=15)
        if confidences:
            mean_conf = np.mean(confidences)
            ax2.axvline(mean_conf, color='red', linestyle='--', linewidth=2,
                        label=f'平均值: {mean_conf:.1f}%')
            ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3: 各类别分布饼图
        type_counts = Counter(types)
        if type_counts:
            colors = [self.colors.get(typ, '#999999') for typ in type_counts.keys()]
            wedges, texts, autotexts = ax3.pie(type_counts.values(), labels=type_counts.keys(),
                                               autopct='%1.1f%%', startangle=90, colors=colors)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax3.set_title('预测类别分布', fontsize=14, fontweight='bold', pad=15)

        # 子图4: 置信度vs证据数量散点图
        evidence_counts = [len(result['evidence']) for result in results]
        ax4.scatter(evidence_counts, confidences, alpha=0.6, color='#FF6B6B', s=80, edgecolors='white', linewidth=1)
        ax4.set_xlabel('证据数量', fontsize=12)
        ax4.set_ylabel('置信度 (%)', fontsize=12)
        ax4.set_title('证据数量 vs 置信度', fontsize=14, fontweight='bold', pad=15)

        if evidence_counts and confidences:
            # 添加趋势线
            z = np.polyfit(evidence_counts, confidences, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(evidence_counts), max(evidence_counts), 100)
            ax4.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2,
                     label=f'趋势线 (r={np.corrcoef(evidence_counts, confidences)[0, 1]:.2f})')
            ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 美化所有子图
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#F8F9FA')
            for spine in ax.spines.values():
                spine.set_color('#DDDDDD')
                spine.set_linewidth(1)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_evidence_wordcloud(self, results):
        """绘制美化版证据词云图"""
        try:
            from wordcloud import WordCloud
        except ImportError:
            print("未安装wordcloud库，跳过词云生成")
            return

        # 收集所有证据词
        all_evidence = []
        for result in results:
            all_evidence.extend(result['evidence'])

        if not all_evidence:
            print("没有证据词数据")
            return

        # 计算词频
        word_freq = Counter(all_evidence)

        # 生成词云
        try:
            plt.figure(figsize=(14, 8))

            wc = WordCloud(
                font_path='simhei.ttf',  # 使用中文字体
                width=1000,
                height=600,
                background_color='white',
                max_words=80,
                colormap='viridis',
                contour_width=2,
                contour_color='steelblue',
                relative_scaling=0.5
            ).generate_from_frequencies(word_freq)

            plt.imshow(wc, interpolation='bilinear')
            plt.axis('off')
            plt.title('关键证据词云图', fontsize=20, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/evidence_wordcloud.png', dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            plt.close()
        except Exception as e:
            print(f"词云生成失败: {e}")

    def plot_evidence_statistics(self, results):
        """绘制证据统计图表"""
        if not results:
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('证据提取统计分析', fontsize=18, fontweight='bold', y=0.95)

        # 子图1: 各类别平均证据数量
        type_evidence_count = defaultdict(list)
        for result in results:
            type_evidence_count[result['pred_type']].append(len(result['evidence']))

        avg_evidence = {typ: np.mean(counts) for typ, counts in type_evidence_count.items()}
        if avg_evidence:
            colors = [self.colors.get(typ, '#999999') for typ in avg_evidence.keys()]
            bars = ax1.bar(avg_evidence.keys(), avg_evidence.values(), color=colors, alpha=0.8,
                           edgecolor='white', linewidth=2)
            ax1.set_title('各类别平均证据数量', fontsize=14, fontweight='bold', pad=15)
            ax1.set_ylabel('平均证据数量', fontsize=12)
            ax1.tick_params(axis='x', rotation=45)

            # 在柱子上添加数值
            for bar, value in zip(bars, avg_evidence.values()):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            ax1.grid(True, alpha=0.3)

        # 子图2: TOP15证据词频
        all_evidence = []
        for result in results:
            all_evidence.extend(result['evidence'])

        evidence_freq = Counter(all_evidence)
        top_15 = evidence_freq.most_common(15)

        if top_15:
            words, counts = zip(*top_15)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(words)))
            y_pos = np.arange(len(words))

            bars = ax2.barh(y_pos, counts, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(words, fontsize=11)
            ax2.set_xlabel('出现次数', fontsize=12)
            ax2.set_title('TOP15 证据词频统计', fontsize=14, fontweight='bold', pad=15)

            # 在条形上添加数值
            for i, (bar, count) in enumerate(zip(bars, counts)):
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                         str(count), ha='left', va='center', fontweight='bold')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3)

        # 子图3: 证据数量分布
        evidence_counts = [len(result['evidence']) for result in results]
        ax3.hist(evidence_counts, bins=10, alpha=0.7, color='#45B7D1',
                 edgecolor='white', linewidth=1.5)
        ax3.set_xlabel('证据数量', fontsize=12)
        ax3.set_ylabel('频次', fontsize=12)
        ax3.set_title('证据数量分布', fontsize=14, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3)

        # 子图4: 各类别证据词分布
        type_evidence_words = defaultdict(list)
        for result in results:
            type_evidence_words[result['pred_type']].extend(result['evidence'])

        # 计算每个类别的独特证据词数量
        unique_evidence_counts = {typ: len(set(words)) for typ, words in type_evidence_words.items()}
        if unique_evidence_counts:
            colors = [self.colors.get(typ, '#999999') for typ in unique_evidence_counts.keys()]
            wedges, texts, autotexts = ax4.pie(unique_evidence_counts.values(),
                                               labels=unique_evidence_counts.keys(),
                                               autopct='%1.1f%%', startangle=90, colors=colors)
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax4.set_title('各类别独特证据词分布', fontsize=14, fontweight='bold', pad=15)

        # 美化所有子图
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_facecolor('#F8F9FA')
            for spine in ax.spines.values():
                spine.set_color('#DDDDDD')
                spine.set_linewidth(1)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/evidence_statistics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_dashboard(self, results, extractor, sample_count=3):
        """创建综合仪表板"""
        if not results:
            return

        # 生成所有图表
        print("生成综合可视化仪表板...")
        self.plot_confidence_analysis(results)
        self.plot_evidence_statistics(results)
        self.plot_evidence_wordcloud(results)

        # 为前几个样本生成详细分析 - 分开显示热力图和权重图
        for i in range(min(sample_count, len(results))):
            self.plot_attention_heatmap(extractor, i)
            self.plot_token_weights(extractor, i)

        print("所有可视化图表已生成完成！")


# 9. 批量处理和分析函数
def batch_extract_evidence(extractor, texts, top_k=3, visualize=True):
    """批量提取证据并生成分析报告"""
    print("=" * 80)
    print("电子取证关键证据提取系统")
    print("=" * 80)

    results = []
    detailed_analyses = []

    for i, text in enumerate(texts, 1):
        print(f"\n【测试案例 {i}】")
        print(f"原始文本: {text[:100]}{'...' if len(text) > 100 else ''}")

        # 提取证据和详细分析
        evidence, scam_type, confidence, analysis = extractor.extract_key_evidence(
            text, top_k=top_k, return_analysis=True
        )

        result = {
            'index': i,
            'text_preview': text[:100],
            'evidence': evidence,
            'pred_type': scam_type,
            'confidence': confidence,
            'text_length': len(text),
            'evidence_count': len(evidence)
        }
        results.append(result)
        detailed_analyses.append(analysis)

        # 打印结果
        print(f"预测类型: {scam_type}")
        print(f"可信度: {confidence}%")
        print(f"关键证据: {evidence}")

        # 预警提示
        if confidence < 60:
            print("低可信度预警：建议人工复核！")
        elif confidence > 90:
            print("高可信度：自动判断可靠")

        if len(evidence) == 0:
            print("警告：未提取到关键证据！")

    print("\n" + "=" * 80)
    print("批量处理完成！")
    print(f"总计处理: {len(texts)} 条文本")

    # 生成可视化
    if visualize and results:
        visualizer = ChineseVisualizer()
        visualizer.create_dashboard(results, extractor)

    # 生成文本报告
    generate_text_report(results, detailed_analyses)

    return results, detailed_analyses


def generate_text_report(results, detailed_analyses):
    """生成详细的文本分析报告"""
    report_file = "evidence_analysis_report.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("关键证据提取分析报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("总体统计:\n")
        f.write(f"- 总样本数: {len(results)}\n")

        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        avg_evidence = np.mean([r['evidence_count'] for r in results]) if results else 0
        f.write(f"- 平均置信度: {avg_confidence:.2f}%\n")
        f.write(f"- 平均证据数量: {avg_evidence:.2f}\n\n")

        f.write("类别分布:\n")
        type_counts = Counter([r['pred_type'] for r in results])
        for scam_type, count in type_counts.items():
            percentage = count / len(results) * 100 if results else 0
            f.write(f"- {scam_type}: {count} 条 ({percentage:.1f}%)\n")
        f.write("\n")

        f.write("详细分析:\n")
        f.write("-" * 40 + "\n")

        for i, (result, analysis) in enumerate(zip(results, detailed_analyses)):
            f.write(f"\n案例 {i + 1}:\n")
            f.write(f"   预测类型: {result['pred_type']}\n")
            f.write(f"   置信度: {result['confidence']}%\n")
            f.write(f"   文本长度: {result['text_length']} 字符\n")
            f.write(f"   关键证据: {', '.join(result['evidence'])}\n")
            f.write(f"   概率分布: {analysis['probabilities']}\n")
            f.write(f"   敏感词匹配: {analysis['sensitive_words_found']}\n")
            f.write(f"   原始文本: {analysis['original_text'][:150]}...\n")

    print(f"详细分析报告已保存至: {report_file}")


# 10. 主函数
def main():
    """主执行函数"""
    try:
        # 模型路径配置
        model_path = r"D:\Projects\Python\Fraud_Messages_Detection\final_code\mini_bert_scam_model\best_model"
        tokenizer_path = r"D:\Projects\Python\Fraud_Messages_Detection\final_code\mini_bert_scam_model\best_tokenizer"

        print("启动关键证据提取系统...")
        print(f"模型路径: {model_path}")
        print(f"Tokenizer路径: {tokenizer_path}")

        # 加载模型
        tokenizer, model = load_bert_model(model_path, tokenizer_path)

        # 创建证据提取器
        extractor = EvidenceExtractor(tokenizer, model, device)

        # 测试数据
        print("\n加载测试数据...")
        try:
            df = pd.read_csv("data.csv")
            test_texts = df['content'].head(800).tolist()
            print(f"成功加载 {len(test_texts)} 条测试文本")
        except Exception as e:
            print(f"无法加载测试数据: {e}")
            print("使用示例数据进行测试...")
            test_texts = [
                "您好，我是快递客服，您的快递丢失了，需要您提供验证码进行退款处理",
                "刷单兼职，日赚300元，需要垫付本金，完成任务后立即返现",
                "投资理财高收益，年化收益率20%，稳赚不赔，内部消息",
                "请问这个商品什么时候发货？物流信息怎么查询？",
                "您的账户存在风险，需要验证身份，请点击链接进行操作"
            ]

        # 批量提取证据
        results, analyses = batch_extract_evidence(extractor, test_texts, top_k=3, visualize=True)

        # 最终总结
        print("\n证据提取完成！")
        print(f"生成的可视化图表保存在 'evidence_visualizations' 目录")
        print(f"详细分析报告保存在 'evidence_analysis_report.txt'")

        # 显示一些统计信息
        if results:
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"平均置信度: {avg_confidence:.1f}%")

            type_dist = Counter([r['pred_type'] for r in results])
            print(f"预测分布: {dict(type_dist)}")

    except Exception as e:
        print(f"系统执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()