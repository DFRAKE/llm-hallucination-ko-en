"""
LLM Hallucination Rate Visualization (EN Single Data)
Fig. 1 - Line chart by domain across turns
Fig. 2 - Heatmap of domain × turn hallucination rates
"""

import matplotlib
matplotlib.use('Agg')  # 디스플레이 없는 환경에서도 실행 가능
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── 데이터 ──────────────────────────────────────────────────────────
turns = ['T1', 'T2', 'T3']

data = {
    'Type A\n(Flutter/Dart)':     [24, 20, 36],
    'Type B\n(C/C++ System)':     [20, 16, 20],
    'Type C\n(KR Certification)': [88, 80, 84],
    'Type D\n(Network Protocol)': [16, 16, 12],
    'Overall':                    [37, 33, 38],
}

colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
markers = ['o', 's', '^', 'D', 'P']

# ── Fig. 1: 도메인별 꺾은선 그래프 ──────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(8, 5))

for (label, values), color, marker in zip(data.items(), colors, markers):
    ax1.plot(turns, values, marker=marker, color=color, linewidth=2,
             markersize=8, label=label.replace('\n', ' '))

ax1.set_title('Fig. 1  Hallucination Rate by Domain Across Turns (EN)',
              fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel('Turn', fontsize=11)
ax1.set_ylabel('Hallucination Rate (%)', fontsize=11)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}%'))
ax1.set_ylim(0, 100)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.85)
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.savefig('Fig1_turn_line_chart.png', dpi=300, bbox_inches='tight')
plt.close()
print('Fig1_turn_line_chart.png 저장 완료')

# ── Fig. 2: 도메인 × 턴 히트맵 ──────────────────────────────────────
domain_labels = [
    'Type A (Flutter/Dart)',
    'Type B (C/C++ System)',
    'Type C (KR Certification)',
    'Type D (Network Protocol)',
    'Overall',
]
matrix = np.array([values for values in data.values()], dtype=float)

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
im = ax2.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

# 컬러바
cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
cbar.set_label('Hallucination Rate (%)', fontsize=10)
cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}%'))

# 축 레이블
ax2.set_xticks(range(len(turns)))
ax2.set_xticklabels(turns, fontsize=11)
ax2.set_yticks(range(len(domain_labels)))
ax2.set_yticklabels(domain_labels, fontsize=10)
ax2.set_title('Fig. 2  Hallucination Rate Heatmap (Domain × Turn, EN)',
              fontsize=12, fontweight='bold', pad=12)

# 셀 내 수치 표기
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = int(matrix[i, j])
        text_color = 'white' if matrix[i, j] > 55 else 'black'
        ax2.text(j, i, f'{val}%', ha='center', va='center',
                 fontsize=11, fontweight='bold', color=text_color)

plt.tight_layout()
plt.savefig('Fig2_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print('Fig2_heatmap.png 저장 완료')
