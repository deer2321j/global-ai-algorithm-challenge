# 📚 基于仲裁式双层集成的智能借阅推荐系统
# (Arbitration-based Two-Layer Ensemble Recommendation System)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Architecture](https://img.shields.io/badge/Arch-ResNet--Inspired-purple?style=for-the-badge)
![Strategy](https://img.shields.io/badge/Strategy-Weighted%20Arbitration-red?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)

> 🏆 **决赛核心方案**: 本项目针对跨模态数据融合、用户冷启动及多模型集成中“概率分布不一致”的核心挑战，创新性地提出了**“加权投票为主、顺序仲裁为辅”**的双层集成框架。系统引入 **ResNet 残差思想** 解决集成退化问题，通过 Layer 1 的自校正与 Layer 2 的抗退化仲裁，最终实现高精度的图书借阅推荐。

---

## 📖 目录 (Table of Contents)

- [核心创新 (Key Innovations)](#-核心创新-key-innovations)
- [系统架构 (System Architecture)](#-系统架构-system-architecture)
- [技术深度解析 (Technical Deep Dive)](#-技术深度解析-technical-deep-dive)
    - [Layer 1: 自校正与概率校准 (Self-Calibration)](#1-layer-1-自校正与概率校准-self-calibration)
    - [Layer 2: 抗退化仲裁决策 (Anti-Degradation Arbitration)](#2-layer-2-抗退化仲裁决策-anti-degradation-arbitration)
    - [跨模型概率可比性校准 (Cross-Model Calibration)](#3-跨模型概率可比性校准-cross-model-calibration)
- [快速开始 (Quick Start)](#-快速开始-quick-start)
- [仓库结构 (Repository Layout)](#-仓库结构-repository-layout)
- [完整复现 (Full Reproduction)](#-完整复现-full-reproduction)
- [注意事项 (Notes)](#-注意事项-notes)

---

## ✨ 核心创新 (Key Innovations)

### 1. 🛡️ 双层仲裁集成架构 (Two-Layer Arbitration Ensemble)
*   **Layer 1 (概率校准与自校正)**: 负责输出高质量的子模型。针对同系列模型（如 `rv5` 不同参数版本）进行内部自洽性验证，提取“不变的稳定部分”赋予高权重，消除随机扰动。
*   **Layer 2 (加权投票与仲裁)**: 作为最终决策层。通过加权投票为每个用户选出唯一且稳定的 Top-1 推荐。引入 **ResNet 跳连思想**，新模型以小权重接入，保证系统性能**单调不减**。

### 2. ⚖️ 跨模型概率可比性校准 (Probability Comparability)
*   **痛点**: 异构模型（如 GNN vs GBDT）输出的概率分布差异巨大，直接融合会导致高置信度模型（即使是错误的）掩盖其他有效信号。
*   **解决方案**:
    *   **文件内标准化**: 消除模型内部的尺度差异。
    *   **跨模型权重融合**: 仅基于验证集性能计算投票权重，**忽略单一模型的原始置信概率**，确保所有模型在公平的跑道上竞争，充分吸收互补性。

---

## 🛠️ 系统架构 (System Architecture)

本方案采用“自校正 -> 全局仲裁”的流式架构，确保数据流从特征提取到最终决策的每一环都具备鲁棒性：

```mermaid
graph TD
    %% =======================
    %% 🎨 样式定义
    %% =======================
    classDef base fill:#f8f9fa,stroke:#adb5bd,stroke-width:1px,color:#212529;
    classDef feature fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,color:#0d47a1;
    classDef layer1 fill:#fff3e0,stroke:#fb8c00,stroke-width:2px,color:#e65100;
    classDef layer2 fill:#fce4ec,stroke:#d81b60,stroke-width:2px,color:#880e4f;
    classDef final fill:#263238,stroke:#263238,stroke-width:3px,color:#fff;

    %% =======================
    %% 1. 输入与特征
    %% =======================
    subgraph Input ["🔍 输入与多模态特征"]
        Data[("📚 原始数据")]:::base
        Feat[("⚡ BERT + GAT + 统计特征")]:::feature
        Data --> Feat
    end

    %% =======================
    %% 2. 模型变体生成
    %% =======================
    subgraph Variants ["🧠 模型变体生成"]
        M_RV5_A[rv5 (Dim=64)]:::base
        M_RV5_B[rv5 (Dim=128)]:::base
        M_Other[其他异构模型]:::base
    end
    Feat --> M_RV5_A & M_RV5_B & M_Other

    %% =======================
    %% 3. Layer 1: 自校正
    %% =======================
    subgraph L1 ["🛡️ Layer 1: 概率校准与自校正"]
        M_RV5_A & M_RV5_B -->|提取共性| SelfCorr[("自校正聚合 (Self-Correction)")]:::layer1
        SelfCorr -->|高权重稳定输出| Stable_RV5[稳定 rv5 信号]:::layer1
    end

    %% =======================
    %% 4. Layer 2: 最终仲裁
    %% =======================
    subgraph L2 ["⚡ Layer 2: 抗退化仲裁决策"]
        Stable_RV5 & M_Other -->|文件内标准化| Norm[标准化处理]:::base
        Norm -->|ResNet式跳连| Vote[("加权投票 (Weighted Voting)")]:::layer2
        Vote -->|得分并列| Arbiter{{"⚖️ 顺序仲裁器"}}:::final
    end

    %% =======================
    %% 5. 输出
    %% =======================
    Result((submission.csv)):::final
    Arbiter --> Result
```

---

## 🔍 技术深度解析 (Technical Deep Dive)

### 1. Layer 1: 自校正与概率校准 (Self-Calibration)
在实验中我们发现，部分深度学习模型（如 `rv5` 系列）对超参数（如文本嵌入维度）敏感，预测结果存在波动。
*   **自纠正机制**: 我们不依赖单一参数设置，而是训练多个变体（如不同 Embedding 维度）。
*   **稳定提取**: 将这些变体在 Layer 1 进行“自纠正”——只有在多个变体中都存在的推荐（共性部分）才会被赋予高权重并传递给下一层。
*   **价值**: 这种机制找到了不断变化中的“稳定不动点”，为 Layer 2 提供了极高置信度的输入。

### 2. Layer 2: 抗退化仲裁决策 (Anti-Degradation Arbitration)
作为最终决策层，Layer 2 的设计核心是**“抗退化”**，类似于 ResNet 的残差结构。
*   **稳定提升逻辑**: 差异化模型提供“补充发现”。
    *   **好样本**: 因多模型权重叠加被筛选出来。
    *   **差样本**: 因权重小或缺乏多数支持，最终被过滤。
*   **ResNet 思想**: 新加入的模型（往往是实验性的）以**小权重**进入体系，类似于 ResNet 的跳连结构（Shortcut）。
    *   如果新模型无效，其微小的权重不会破坏主干（性能不退化）。
    *   只有当新模型在某些特定 User-Book 对上提供了显著的正确增益时，才会提升总分。
    *   **结果**: 保证了集成效果的**单调不减**，让我们敢于引入更多差异化模型。

### 3. 跨模型概率可比性校准 (Cross-Model Calibration)
为了解决异构模型概率分布不可比的问题（例如 GNN 输出倾向于 0.9，而 GBDT 倾向于 0.6），我们采取了激进的校准策略：
*   **文件内标准化 (In-file Standardization)**: 强制将每个子模型的输出拉伸到统一尺度，消除量纲差异。
*   **基于权重的纯投票**: 在最终融合时，**不再参考**模型输出的原始概率值，而是完全依赖该模型在验证集上的**权重**进行投票。
    *   这确保了模型之间是“公平竞争”的，避免了某个“过自信”的模型主导投票，显著提升了系统的泛化能力与鲁棒性。

---

## 🚀 快速开始 (Quick Start)

在 **项目根目录** 下运行以下命令即可生成最终提交文件：

```bash
python FINAL加权.py
```

*   **输出**: `submission.csv`
*   **注意**: 脚本依赖各子目录的中间产物，如果是首次运行，请参考下方的“完整复现”。

---

## 📂 仓库结构 (Repository Layout)

```text
.
├── 📜 FINAL加权.py                # 🔥 Layer 2 核心：最终仲裁与加权脚本 -> submission.csv
├── 📜 Top10加权融合.py            # 🛡️ Layer 1 核心：Top-10 基准生成 -> top10加权输出结果.csv
├── 📜 整合rv5到最终投票.py         # 🔧 Layer 1 核心：rv5 自校正与标准化 -> 七以上的v5.csv
├── 📂 23混推/                     # 🧠 子模型：混合推荐策略
├── 📂 v5/                        # 🧠 子模型：v5 系列 (含自校正逻辑)
├── 📂 dspos2/                    # 🧠 子模型：dspos2 版本
├── 📂 133/                       # 🧠 子模型：133 版本
├── 📂 f1/                        # 🧠 子模型：f1 版本
├── 📂 v2/                        # 🧠 子模型：v2 版本
├── 📂 决赛classic_autoML/        # 🧠 子模型：AutoML 策略
└── 📝 环境依赖.txt                 # 📦 依赖说明
```

---

## 🔄 完整复现 (Full Reproduction)

### 1. 子模型生成 (Per-model Inference)
进入以下每个文件夹，按照其内部 README 运行，生成各自的 CSV 产物：
> `23混推` / `v5` / `dspos2` / `133` / `f1` / `v2` / `决赛classic_autoML`

### 2. 执行双层集成 (Execute Ensemble)

```bash
# Step 1: Layer 1 自校正 (Self-Correction)
# 对 rv5 系列进行内部加权与标准化，提取稳定信号
python 整合rv5到最终投票.py

# Step 2: 准备基准数据
# 生成 Top-10 辅助基准
python Top10加权融合.py

# Step 3: Layer 2 最终仲裁 (Final Arbitration)
# 全量模型标准化 + 加权投票 + 顺序仲裁
python FINAL加权.py
```

---

## ⚠️ 注意事项 (Notes)

*   **数据路径**: 请确保 `1data.csv` (复赛) 和 `111data.csv` (决赛) 放置在正确位置。
*   **运行目录**: 所有 Python 脚本必须在 **项目根目录** 执行。
*   **大文件**: 模型权重文件 (`.pkl`, `.joblib`) 未上传 Git，如需完整离线包请联系 `a1992423911@dlmu.edu.cn`。

---

<p align="center">
  <i>Powered by <b>Arbitration-based Ensemble Strategy</b></i>
</p>
