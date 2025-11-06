import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
import os
import re
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')


# 设置中文字体，解决中文显示问题
def setup_chinese_font():
    """设置中文字体，解决中文显示问题（优化版）"""
    print("\n🔤 开始配置中文字体...")
    # 定义常见中文字体名称和可能的系统路径
    chinese_font_names = [
        'SimHei', 'Microsoft YaHei', 'Microsoft YaHei UI',
        'PingFang SC', 'Songti SC', 'KaiTi SC',
        'WenQuanYi Zen Hei', 'DejaVu Sans'
    ]

    # 常见中文字体文件路径
    common_font_paths = [
        'C:/Windows/Fonts/simhei.ttf',  # Windows
        'C:/Windows/Fonts/msyh.ttc',  # Windows
        'C:/Windows/Fonts/msyhl.ttc',  # Windows
        '/Library/Fonts/PingFang SC.ttc',  # macOS
        '/usr/share/fonts/wenquanyi/wqy-zenhei/wqy-zenhei.ttc'  # Linux
    ]

    try:
        import matplotlib.font_manager as fm
        # 先尝试通过字体名称设置
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        target_font = None

        # 查找可用的中文字体
        for font_name in chinese_font_names:
            if font_name in available_fonts:
                target_font = font_name
                break

        if target_font:
            plt.rcParams['font.sans-serif'] = [target_font] + ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            print(f"✅ 成功使用系统字体: {target_font}")
            return

        # 若未找到注册字体，尝试手动加载字体文件
        for font_path in common_font_paths:
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()

                plt.rcParams['font.sans-serif'] = [font_name] + ['DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 手动加载字体成功: {font_name} (路径: {font_path})")
                return

        # 所有方法失败，使用默认字体并提示
        print("⚠️ 未找到任何中文字体，将使用默认字体（中文可能显示为方框）")
        print("💡 建议手动安装字体：")
        print("   - Windows: 安装 SimHei（黑体）或 Microsoft YaHei（微软雅黑）")
        print("   - macOS: 确保 PingFang SC（苹方）已启用")
        print("   - Linux: 安装 WenQuanYi Zen Hei（文泉驿正黑）")

    except Exception as e:
        print(f"❌ 字体设置过程出错: {str(e)}")
        # 出错后强制设置基础参数
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False


# 初始化字体
setup_chinese_font()
sns.set_style("whitegrid")

# 诈骗类型配置
LABEL_MAP = {0: "正常对话", 1: "刷单返利", 2: "虚假投资", 3: "冒充客服"}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# 大类映射：正常 vs 诈骗
BINARY_MAP = {
    "正常对话": "正常",
    "刷单返利": "诈骗",
    "虚假投资": "诈骗",
    "冒充客服": "诈骗"
}


class HumanVsModelFinal:
    def __init__(self, model_path, tokenizer_path):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用计算设备: {self.device}")

    def load_model(self):
        """加载模型"""
        print("\n📥 开始加载模型...")
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_path)
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=len(LABEL_MAP),
                ignore_mismatched_sizes=True
            )
            self.model.eval()
            self.model.to(self.device)
            print("✅ 模型加载成功！")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def model_predict(self, text):
        """模型预测"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            pred_label = np.argmax(probs)
            pred_category = LABEL_MAP[pred_label]
            confidence = round(probs[pred_label] * 100, 2)

            return pred_label, pred_category, confidence

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return -1, "预测错误", 0

    def load_data_with_true_labels(self):
        """加载带真实标签的数据"""
        print("\n📁 正在加载带标签的数据...")
        data_files = ["data.csv", "preprocessed_scam_data.csv"]

        for file in data_files:
            if os.path.exists(file):
                try:
                    df = pd.read_csv(file)
                    if 'content' not in df.columns or 'label' not in df.columns:
                        print(f"⚠️ {file} 缺少content或label列")
                        continue

                    df = df.dropna(subset=['content', 'label'])
                    df = df[df['content'].str.len() >= 5]
                    df['label'] = df['label'].astype(int)

                    valid_labels = set(LABEL_MAP.keys())
                    df = df[df['label'].isin(valid_labels)]

                    if len(df) > 0:
                        print(f"✅ 从 {file} 加载数据: {len(df)} 条带标签样本")
                        return df

                except Exception as e:
                    print(f"❌ 加载 {file} 失败: {e}")

        # 使用示例数据
        print("⚠️ 未找到带标签数据文件，使用示例数据")
        example_data = [
            {"content": "您好，我是快递客服，您的快递丢失了，需要您提供验证码进行退款处理", "label": 3},
            {"content": "刷单兼职，日赚300元，需要垫付本金，完成任务后立即返现", "label": 1},
            {"content": "投资理财高收益，年化收益率20%，稳赚不赔，内部消息", "label": 2},
            {"content": "请问这个商品什么时候发货？物流信息怎么查询？", "label": 0},
            {"content": "您的账户存在风险，需要验证身份，请点击链接进行操作", "label": 3},
            {"content": "点赞关注抖音账号，每条2元，日结工资，无需押金", "label": 1},
            {"content": "银行贷款，额度20万，利率优惠，快速放款", "label": 2},
            {"content": "淘宝客服通知：您的订单异常，需要重新确认支付信息", "label": 3},
            {"content": "这个产品的质量怎么样？有优惠活动吗？", "label": 0},
            {"content": "股票投资群，老师带单，保证盈利，加群领取牛股", "label": 2}
        ]
        return pd.DataFrame(example_data)

    def human_labeling_session(self, texts, true_labels):
        """人工标注环节"""
        print("\n" + "=" * 60)
        print("🧑‍💻 人工标注环节")
        print("=" * 60)
        print("0:正常对话 1:刷单返利 2:虚假投资 3:冒充客服")
        print("输入q退出，输入r重新查看当前文本")
        print("=" * 60)

        human_results = []
        total_human_time = 0

        for i, (text, true_label) in enumerate(zip(texts, true_labels), 1):
            true_category = LABEL_MAP[true_label]

            print(f"\n样本 {i}/{len(texts)}:")
            print(f"原文: {text[:80]}{'...' if len(text) > 80 else ''}")

            start_time = time.time()
            while True:
                try:
                    user_input = input("请输入分类(0-3): ").strip()

                    if user_input.lower() == 'q':
                        return human_results
                    if user_input.lower() == 'r':
                        print(f"重新显示: {text[:80]}{'...' if len(text) > 80 else ''}")
                        continue

                    human_label = int(user_input)
                    if human_label not in LABEL_MAP.keys():
                        print("❌ 请输入0-3")
                        continue

                    process_time = time.time() - start_time
                    total_human_time += process_time

                    human_category = LABEL_MAP[human_label]
                    human_binary = BINARY_MAP[human_category]
                    true_binary = BINARY_MAP[true_category]

                    human_binary_correct = (human_binary == true_binary)
                    human_detailed_correct = (human_label == true_label)

                    result = {
                        'text': text,
                        'true_label': true_label,
                        'true_category': true_category,
                        'true_binary': true_binary,
                        'human_label': human_label,
                        'human_category': human_category,
                        'human_binary': human_binary,
                        'human_time': round(process_time, 3),
                        'human_binary_correct': human_binary_correct,
                        'human_detailed_correct': human_detailed_correct
                    }
                    human_results.append(result)

                    correct_mark = "✅" if human_detailed_correct else "❌"
                    print(f"标注: {human_category} {correct_mark} (真实: {true_category})")
                    break

                except ValueError:
                    print("❌ 输入无效")
                except KeyboardInterrupt:
                    return human_results

        return human_results

    def run_model_predictions(self, human_results):
        """运行模型预测"""
        print("\n" + "=" * 60)
        print("🤖 模型预测环节")
        print("=" * 60)

        model_results = []
        total_model_time = 0

        for i, human_result in enumerate(human_results, 1):
            print(f"\n进度: {i}/{len(human_results)}")

            start_time = time.time()
            pred_label, pred_category, confidence = self.model_predict(human_result['text'])
            process_time = time.time() - start_time
            total_model_time += process_time

            pred_binary = BINARY_MAP[pred_category] if pred_label != -1 else "错误"
            model_binary_correct = (pred_binary == human_result['true_binary']) if pred_label != -1 else False
            model_detailed_correct = (pred_label == human_result['true_label']) if pred_label != -1 else False

            model_result = human_result.copy()
            model_result.update({
                'model_label': pred_label,
                'model_category': pred_category,
                'model_binary': pred_binary,
                'model_confidence': confidence,
                'model_time': round(process_time, 3),
                'model_binary_correct': model_binary_correct,
                'model_detailed_correct': model_detailed_correct
            })
            model_results.append(model_result)

            human_correct = "✅" if human_result['human_detailed_correct'] else "❌"
            model_correct = "✅" if model_detailed_correct else "❌"
            print(f"真实: {human_result['true_category']}")
            print(f"人工: {human_result['human_category']} {human_correct}")
            print(f"模型: {pred_category} {model_correct} ({confidence}%)")

        return model_results

    def calculate_comparison_metrics(self, results):
        """计算对比指标"""
        print("\n" + "=" * 60)
        print("📊 准确率统计结果")
        print("=" * 60)

        if not results:
            return {}

        total_samples = len(results)

        # 人工准确率
        human_binary_correct = sum(1 for r in results if r['human_binary_correct'])
        human_detailed_correct = sum(1 for r in results if r['human_detailed_correct'])
        human_binary_accuracy = (human_binary_correct / total_samples) * 100
        human_detailed_accuracy = (human_detailed_correct / total_samples) * 100

        # 模型准确率
        model_binary_correct = sum(1 for r in results if r['model_binary_correct'])
        model_detailed_correct = sum(1 for r in results if r['model_detailed_correct'])
        model_binary_accuracy = (model_binary_correct / total_samples) * 100
        model_detailed_accuracy = (model_detailed_correct / total_samples) * 100

        # 时间统计
        human_times = [r['human_time'] for r in results]
        model_times = [r['model_time'] for r in results]
        avg_human_time = np.mean(human_times)
        avg_model_time = np.mean(model_times)
        speedup_ratio = avg_human_time / avg_model_time if avg_model_time > 0 else 0

        # 各类别准确率
        human_category_stats = {}
        model_category_stats = {}
        for category in LABEL_MAP.values():
            human_category_samples = [r for r in results if r['true_category'] == category]
            if human_category_samples:
                human_correct = sum(1 for r in human_category_samples if r['human_detailed_correct'])
                model_correct = sum(1 for r in human_category_samples if r['model_detailed_correct'])

                human_category_stats[category] = {
                    'count': len(human_category_samples),
                    'correct': human_correct,
                    'accuracy': (human_correct / len(human_category_samples)) * 100
                }
                model_category_stats[category] = {
                    'count': len(human_category_samples),
                    'correct': model_correct,
                    'accuracy': (model_correct / len(human_category_samples)) * 100
                }

        # 打印结果
        print(f"样本总数: {total_samples}")
        print(f"\n🎯 大类准确率 (正常vs诈骗):")
        print(f"  人工: {human_binary_accuracy:.1f}% ({human_binary_correct}/{total_samples})")
        print(f"  模型: {model_binary_accuracy:.1f}% ({model_binary_correct}/{total_samples})")

        print(f"\n🎯 小类准确率 (四分类):")
        print(f"  人工: {human_detailed_accuracy:.1f}% ({human_detailed_correct}/{total_samples})")
        print(f"  模型: {model_detailed_accuracy:.1f}% ({model_detailed_correct}/{total_samples})")

        print(f"\n⏱️  效率对比:")
        print(f"  人工平均用时: {avg_human_time:.2f}秒/条")
        print(f"  模型平均用时: {avg_model_time:.3f}秒/条")
        print(f"  加速比: {speedup_ratio:.1f}倍")

        return {
            'total_samples': total_samples,
            'human_binary_accuracy': human_binary_accuracy,
            'human_detailed_accuracy': human_detailed_accuracy,
            'model_binary_accuracy': model_binary_accuracy,
            'model_detailed_accuracy': model_detailed_accuracy,
            'avg_human_time': avg_human_time,
            'avg_model_time': avg_model_time,
            'speedup_ratio': speedup_ratio,
            'human_category_stats': human_category_stats,
            'model_category_stats': model_category_stats,
            'all_results': results
        }

    def plot_final_charts(self, stats):
        """绘制最终图表（强制指定中文字体）"""
        os.makedirs("efficiency_test_result", exist_ok=True)

        # 关键：强制设置字体参数
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 再次明确指定黑体
        plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示
        plt.rcParams['font.size'] = 10  # 调整基础字体大小

        # 图表1: 准确率对比
        plt.figure(figsize=(10, 6))

        categories = ['大类准确率', '小类准确率']
        human_acc = [stats['human_binary_accuracy'], stats['human_detailed_accuracy']]
        model_acc = [stats['model_binary_accuracy'], stats['model_detailed_accuracy']]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = plt.bar(x - width / 2, human_acc, width, label='人工', color='#2E8B57', alpha=0.8)
        bars2 = plt.bar(x + width / 2, model_acc, width, label='模型', color='#4682B4', alpha=0.8)

        # 每个文本都手动指定字体
        plt.ylabel('准确率 (%)', fontproperties='SimHei', fontsize=12)
        plt.title('人工 vs 模型 准确率对比', fontproperties='SimHei', fontsize=14, fontweight='bold')
        plt.xticks(x, categories, fontproperties='SimHei', fontsize=11)
        plt.legend(prop={'family': 'SimHei', 'size': 11})  # 图例字体单独设置
        plt.ylim(0, 105)

        # 添加数值标签（指定字体）
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                         f'{height:.1f}%', ha='center', va='bottom',
                         fontweight='bold', fontproperties='SimHei', fontsize=10)

        plt.tight_layout()
        plt.savefig('efficiency_test_result/accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 图表2: 效率对比
        plt.figure(figsize=(8, 6))

        labels = ['人工标注', '模型预测']
        times = [stats['avg_human_time'], stats['avg_model_time']]
        colors = ['#FFA07A', '#20B2AA']

        bars = plt.bar(labels, times, color=colors, alpha=0.8)

        # 每个文本手动指定字体
        plt.ylabel('处理时间 (秒)', fontproperties='SimHei', fontsize=12)
        plt.title('处理效率对比', fontproperties='SimHei', fontsize=14, fontweight='bold')
        plt.xticks(fontproperties='SimHei', fontsize=11)

        # 添加数值标签
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{time_val:.3f}s', ha='center', va='bottom',
                     fontweight='bold', fontproperties='SimHei', fontsize=10)

        plt.tight_layout()
        plt.savefig('efficiency_test_result/efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ 图表已生成: efficiency_test_result/accuracy_comparison.png")
        print("✅ 图表已生成: efficiency_test_result/efficiency_comparison.png")

    def generate_final_report(self, results, stats):
        """生成最终报告（包含原文）"""
        report_path = 'efficiency_test_result/detailed_comparison_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("人工 vs 模型 对比评估报告\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # 总体统计
            f.write("一、总体统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"样本总数: {stats['total_samples']}\n")
            f.write(
                f"大类准确率: 人工{stats['human_binary_accuracy']:.1f}% vs 模型{stats['model_binary_accuracy']:.1f}%\n")
            f.write(
                f"小类准确率: 人工{stats['human_detailed_accuracy']:.1f}% vs 模型{stats['model_detailed_accuracy']:.1f}%\n")
            f.write(
                f"处理效率: 人工{stats['avg_human_time']:.2f}s vs 模型{stats['avg_model_time']:.3f}s (加速{stats['speedup_ratio']:.1f}倍)\n\n")

            # 各类别统计
            f.write("二、各类别准确率\n")
            f.write("-" * 40 + "\n")
            for category in LABEL_MAP.values():
                if category in stats['human_category_stats']:
                    human_stats = stats['human_category_stats'][category]
                    model_stats = stats['model_category_stats'][category]
                    diff = model_stats['accuracy'] - human_stats['accuracy']
                    f.write(
                        f"{category}: 人工{human_stats['accuracy']:.1f}% vs 模型{model_stats['accuracy']:.1f}% (差异{diff:+.1f}%)\n")
            f.write("\n")

            # 详细样本数据（包含原文）
            f.write("三、详细样本数据\n")
            f.write("-" * 120 + "\n")
            header = f"{'序号':<3} {'真实类别':<8} {'人工标注':<8} {'模型预测':<8} {'人工正确':<6} {'模型正确':<6} {'原文摘要':<40}\n"
            f.write(header)
            f.write("-" * 120 + "\n")

            for i, r in enumerate(results, 1):
                human_correct = "✅" if r['human_detailed_correct'] else "❌"
                model_correct = "✅" if r['model_detailed_correct'] else "❌"
                text_preview = r['text'][:35] + "..." if len(r['text']) > 35 else r['text']

                line = f"{i:<3} {r['true_category']:<8} {r['human_category']:<8} {r['model_category']:<8} {human_correct:<6} {model_correct:<6} {text_preview:<40}\n"
                f.write(line)

            f.write("\n" + "=" * 100 + "\n")
            f.write("四、完整原文内容\n")
            f.write("=" * 100 + "\n")

            for i, r in enumerate(results, 1):
                f.write(f"\n【样本 {i}】\n")
                f.write(f"真实类别: {r['true_category']}\n")
                f.write(f"人工标注: {r['human_category']}\n")
                f.write(f"模型预测: {r['model_category']} (置信度: {r['model_confidence']}%)\n")
                f.write(f"人工用时: {r['human_time']:.2f}s | 模型用时: {r['model_time']:.3f}s\n")
                f.write(f"原文: {r['text']}\n")
                f.write("-" * 80 + "\n")

        print(f"📄 详细报告已保存至: {report_path}")


def main():
    """主函数"""
    # 配置路径
    model_path = "mini_bert_scam_model/best_model"
    tokenizer_path = "mini_bert_scam_model/best_tokenizer"

    # 初始化评估系统
    evaluator = HumanVsModelFinal(model_path, tokenizer_path)

    # 加载模型
    if not evaluator.load_model():
        return

    # 加载数据
    df = evaluator.load_data_with_true_labels()
    sample_size = min(10, len(df))

    test_data = df.sample(n=sample_size, random_state=42)
    texts = test_data['content'].tolist()
    true_labels = test_data['label'].tolist()

    print(f"\n测试样本: {sample_size}条")

    # 人工标注
    human_results = evaluator.human_labeling_session(texts, true_labels)
    if not human_results:
        print("❌ 人工标注未完成")
        return

    # 模型预测
    model_results = evaluator.run_model_predictions(human_results)

    # 统计分析
    stats = evaluator.calculate_comparison_metrics(model_results)

    # 生成图表和报告
    evaluator.plot_final_charts(stats)
    evaluator.generate_final_report(model_results, stats)

    print("\n" + "=" * 60)
    print("🎉 评估完成！")
    print("📊 2个图表: efficiency_test_result/ 目录")
    print("📄 详细报告: efficiency_test_result/detailed_comparison_report.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
