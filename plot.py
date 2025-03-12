import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.stats import linregress

# データをCSVファイルから読み込み
data = pd.read_csv('./data/output.csv', sep=',')

# Matched ScoreごとにCosine Similarityをグループ化
grouped_data = data.groupby('Matched Score')['Cosine Similarity'].apply(list)

# 箱ひげ図の描画
plt.rcParams["font.size"] = 24
plt.figure(figsize=(16, 12))
plt.boxplot(grouped_data, labels=grouped_data.index, vert=True)
plt.xlabel('Matched Score')
plt.ylabel('コサイン類似度')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 描画
plt.show()

# 回帰直線のプロットと相関係数の計算
plt.figure(figsize=(16, 12))

# 散布図をプロット
matched_scores = data['Matched Score']
cosine_similarities = data['Cosine Similarity']
plt.scatter(matched_scores, cosine_similarities, alpha=0.5, label='データポイント')

# 回帰直線を計算
slope, intercept, r_value, p_value, std_err = linregress(matched_scores, cosine_similarities)

# 回帰直線をプロット
x = np.linspace(matched_scores.min(), matched_scores.max(), 100)
y = slope * x + intercept
plt.plot(x, y, color='red', label=f'回帰直線 (y = {slope:.2f}x + {intercept:.2f})')

# 相関係数を表示
plt.text(
    0.05, 0.95, f"相関係数: {r_value:.2f}", fontsize=18, transform=plt.gca().transAxes, 
    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white", alpha=0.7)
)

# グラフの装飾
plt.xlabel('Matched Score')
plt.ylabel('コサイン類似度')
plt.legend()
plt.grid(axis='both', linestyle='--', alpha=0.7)

# 描画
plt.show()

# Matched Scoreごとの個数を計算
matched_score_counts = data['Matched Score'].value_counts().sort_index()

# MS棒グラフの描画
plt.rcParams["font.size"] = 24
plt.figure(figsize=(15, 12))
plt.bar(matched_score_counts.index, matched_score_counts.values, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Matched Score')
plt.ylabel('件数')
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 描画
plt.show()

# Cosine類似度のヒストグラム
plt.rcParams["font.size"] = 24
plt.figure(figsize=(16, 12))
bins = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
labels = ['<0.50', '0.50-0.55', '0.55-0.60', '0.60-0.65', '0.65-0.70', 
          '0.70-0.75', '0.75-0.80', '0.80-0.85', '0.85-0.90', '0.90-0.95', '0.95-1.00']

# Cosine類似度を指定した範囲に分類
data['Cosine Range'] = pd.cut(data['Cosine Similarity'], bins=bins, labels=labels[1:], right=False)
data['Cosine Range'] = data['Cosine Range'].cat.add_categories([labels[0]]).fillna(labels[0])

# 階級ごとの頻度を集計
range_counts = data['Cosine Range'].value_counts().reindex(labels)

# ヒストグラムの描画
plt.bar(range_counts.index, range_counts.values, edgecolor='black', alpha=0.7)

# グラフのラベルとタイトル
plt.xlabel('コサイン類似度')
plt.ylabel('頻度')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
# 描画
plt.show()