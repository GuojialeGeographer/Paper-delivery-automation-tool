# ABMind ABM+细分学科的论文速递项目自动化工作流

## 1. 项目概述 (Project Overview)

本项目旨在推动 Agent-Based Modeling (ABM) 在中文社区的推广，特别关注 ABM 与地理信息系统（后期可以是任意其他学科领域，先以GIS作为测试案例进行开发）的交叉应用。核心任务是从海量学术文献中系统地筛选、整理和解读 ABM+GIS 领域（后期可以是任意其他学科领域）的高质量研究论文，并定期以“ABMind论文速递”的形式发布，重点宣传以 Mesa 为代表的 ABM 工具。

为了提高效率和应对大量文献，本项目规划了自动化流程，特别是利用大型语言模型 API（如 Gemini API）辅助文献的摘要翻译、亮点提取和推荐理由生成。

## 2. 项目目标 (Project Goals)

* 建立并维护一个包含 ABM+GIS学科主题 相关文献的数据库。
* 设计一套高效、可重复的文献筛选和优先排序工作流。
* 定义标准的单篇文献信息提取和解读格式。
* 实现文献摘要翻译、研究亮点和推荐理由生成的自动化或半自动化。
* 定期生成高质量的公众号推文内容（如论文速递）。
* 通过分享前沿研究，促进中文社区对 ABM 和 Mesa 的认知和应用。

## 3. 数据来源 (Data Source)

* **主要来源:** Web of Science (WOS)
* **检索语言:** 中文 (Language: Chinese)
* **检索式 (示例):** TS = ((“agent-based model*”) AND (“Geograph* information system*” OR geograph* OR “Geospatial” OR GIS))
* **初始数据量:** 700+ 篇文章 (截至检索时)
* **数据字段 (原始 CSV):** 作者, 论文标题, 期刊, Abstract, Publication Date, Publication Year, Volume, Issue, DOI, DOI Link 等。
* **后续扩展:** 可能包括 CNKI 等中文数据库，以及相关领域的英文顶刊文献（需注明来源）。

## 4. 工作流 (Workflow)

整个工作流分为以下几个阶段：

### 阶段 1: 文献收集、管理与初步处理 (Collection, Management & Initial Processing),其中这一步将有用户自己去WOS检索导出并作初步的预处理以得到一个合格的CSV文件。

* **目标:** 构建一个结构化的、易于管理的文献库。
* **步骤:**
  1. 执行 WOS 检索，导出文献数据 (CSV 或其他结构化格式)。
  2. 将导出的数据导入文献管理工具 (如 Zotero, EndNote) 或数据处理软件 (如 Excel, Python pandas)。
  3. **数据清洗:** 处理可能存在的格式错误、缺失值。
  4. **字段提取:** 确保关键字段 (标题、作者、期刊、发表日期/年份、DOI、摘要) 被正确识别和提取。
  5. **去重:** 删除重复的文献记录。
  6. **初步分类/标记 (可选):** 根据年份、期刊水平或粗略主题进行初步标记，辅助后续筛选。

### 阶段 2: 文献筛选与优先级排序 (Filtering & Prioritization)

* **目标:** 从文献库中确定本期论文速递的文献列表，并按指定顺序排列。
* **步骤:**
  1. **核心筛选:** 过滤非研究论文类型 (如会议摘要、编辑材料)，确保文献与 ABM+GIS 高度相关 (基于标题和摘要快速判断)。
  2. **优先级排序:** **严格按照发表时间进行降序排列** (优先按年份，其次按月份，最后按日期)。这是确保“速递”新颖性的最关键步骤。
  3. **提取本期列表:** 从排序结果的顶部，连续提取指定数量的文献 (例如，每周 20 篇)。
  4. **状态更新:** 在文献库中标记这些文献为“已处理/已收录于速递 #X”，避免重复选取。

### 阶段 3: 内容信息提取与生成 (Content Extraction & Generation)

* **目标:** 为选定的每篇文献生成符合推文格式的详细内容。
* **步骤 (对本期列表中的每篇文献执行):**
  1. **提取标准信息:** 准确提取英文标题、作者、期刊/来源、DOI、英文摘要。
  2. **翻译与生成:**
     * **中文标题翻译:** 对英文标题进行准确、地道的中文翻译。
     * **中文摘要翻译:** 对英文摘要进行准确、流畅的中文翻译。
     * **关键词提取与翻译:** 从英文标题和摘要中提取核心术语作为关键词 (3-5个)，并提供中英文。
     * **研究亮点生成 (纯中文, 分点):** 基于摘要内容，提炼论文最重要、最突出的 1-3 个贡献、发现或方法创新点。
     * **推荐理由生成 (纯中文, 分点):** **综合**原文献中可能提及的“ABM+GIS融合方式”信息，结合研究亮点以及对 ABmind 社区和 Mesa 推广的价值（例如，方法创新、应用领域有代表性、对 Mesa 用户有启发、适合用 Mesa 复现/拓展等），重新思考并提炼出 2-4 个简洁、有吸引力的推荐理由。这部分需要对文献内容进行理解和重组。
  3. **结构化存储:** 将提取和生成的内容组织成指定的 Markdown 格式。

### 阶段 4: 推文组装、审核与发布 (Assembly, Review & Publishing)

* **目标:** 整合单篇文献内容，形成完整的推文，并进行发布。
* **步骤:**
  1. **组装推文:** 将所有整理好的单篇文献内容按照第 2 阶段确定的顺序 (时间倒序) 拼接起来。
  2. **添加引言和结尾:** 编写符合 ABmind 社区定位的开篇引言和总结性结尾 (参考已有的社区介绍和推文结构)。
  3. **整体排版:** 使用 Markdown 格式进行排版，确保清晰易读。
  4. **内容审核:** 仔细校对所有文本，确保准确性、流畅性和格式正确性。检查 DOI 链接是否有效。
  5. **发布:** 将 Markdown 内容导入微信公众号编辑器或其他发布平台，进行最终美化和发布。
  6. **推广:** 在社区群、网站等渠道进行推广。

## 5. 输出格式规范 (Output Format Specification - Markdown)

以下是单篇文献在推文中的 Markdown 输出格式规范：

```markdown
---

**第 [序号] 篇**

**【论文标题】**
[English Title]
[中文翻译标题]

**【作者】**
[Authors List]

**【期刊/来源】**
[Journal Name], [Year], [Volume]([Issue]) // 或其他包含卷期页码的格式
[DOI Link] // 可点击的 DOI 链接

**【摘要】**
[English Abstract]
[中文翻译摘要]

**【关键词】**
[English Keywords, comma separated]

**【研究亮点】✨**
*   亮点 1 (纯中文)
*   亮点 2 (纯中文)
*   亮点 3 (纯中文)
// 通常 1-3 点

**【推荐理由】👍**
*   理由 1 (纯中文)
*   理由 2 (纯中文)
*   理由 3 (纯中文)
// 通常 2-4 点，综合【ABM+当前学科主题】的融合方式、对这个【ABM+当前学科主题】领域研究的价值、方法创新等

---
// 用于分隔不同论文
```

## 9. 最新自动化输出内容规范与用户需求（2025-04-21 更新）

### 1. 英文信息处理

- 直接使用用户上传的CSV中的英文信息（标题、作者、期刊、摘要、DOI/DOI Link等），不进行多版本或多选项输出。

### 2. 中文翻译与内容生成

- 所有中文输出（标题、摘要、关键词、亮点、推荐理由）均只输出最标准、最符合学术规范的版本。
- 论文标题翻译仅输出一种标准学术风格，不输出多个选项或“推荐翻译”等描述。

### 3. DOI 处理

- 只输出一行 DOI Link（优先用 DOI Link 字段，如无则自动补全为 https://doi.org/xxx）。不再同时输出 DOI 号和 DOI Link。

### 4. 亮点与推荐理由格式

- 亮点和推荐理由直接输出内容，不加“基于该论文摘要，以下是提炼的1-3条最重要的研究亮点：”或“以下是基于论文内容生成的推荐理由，供参考：”等前缀说明。
- 亮点和推荐理由均为纯中文条目，亮点建议1-3条，推荐理由建议2-4条。

### 5. 其他内容规范

- 输出风格直接、精炼，所有内容可直接用于公众号、社群等平台发布，无需用户二次筛选或挑选。
- 自动适配用户上传的CSV字段（如 Title、Author、Journal、Abstract、Publication、DOI、DOI Link 等），并自动处理年份提取和格式化。

### 6. 用户需求变更管理

- 每次需求或输出规范有变动，须及时补充到本README文件“最新自动化输出内容规范与用户需求”部分，确保团队和后续开发者始终对齐。

## 6. 自动化潜力与未来工作 (Automation Potential & Future Work)

本项目旨在逐步自动化上述工作流中的部分或全部环节，以提高效率和可扩展性。

* **自动化目标:**
  * 阶段 1 部分: 批量从 WOS 导出数据，自动导入数据库或结构化文件。
  * 阶段 2 部分: 基于设定的关键词和逻辑进行初步筛选，自动按时间排序和选取本期列表。
  * 阶段 3 (核心自动化): 利用 Gemini API 等大型语言模型，实现：
    * 自动执行中英文标题和摘要翻译。
    * 自动从摘要中提取关键词。
    * 自动从摘要和全文 (如果可获取) 中识别和提炼研究亮点 (ABM、GIS、应用、发现等)。
    * 自动分析文献内容，生成符合要求的、有价值的推荐理由。
  * 阶段 4 部分: 自动将整理好的单篇内容组装成完整的推文 Markdown 文件。
* **技术路线:**
  * 使用 Python 编写脚本，调用 WOS API (如果可用) 或处理导出的数据文件 (CSV, XML)。
  * 使用 Python 调用 Gemini API 进行文本生成和翻译任务。
  * 使用数据库 (如 SQLite) 存储文献信息和处理状态。
  * 开发简单的用户界面或命令行工具来管理工作流和审核自动化结果。
* **挑战:**
  * 确保 AI 生成内容的准确性和质量，特别是对研究亮点和推荐理由的理解和提炼。需要人工审核。
  * 处理不同文献摘要风格和内容复杂性的差异。
  * 如何让 AI 理解并体现“对 ABmind 社区和 Mesa 的价值”这一特定推荐理由。
  * 处理版权问题，特别是当需要访问全文来辅助亮点提取时。
* **未来展望:**
  * 扩展文献来源。
  * 开发更精细的文献分类和聚类功能。
  * 探索 AI 辅助的模型复现或代码理解。

## 7. 框架合理性与自信来源 (Framework Rationality & Confidence)

这份工作流框架是合理的，并且具备实现目标的自信，原因在于：

* **结构化方法:** 将复杂任务分解为清晰的阶段，便于管理和逐步实现自动化。
* **数据驱动:** 流程基于 WOS 的实际文献数据，而非凭空想象。
* **客观排序:** 采用严格的时间排序作为主要标准，保证了内容的新颖性，是客观且可验证的。
* **模块化设计:** 工作流中的各个阶段和步骤相对独立，可以独立开发和测试，也便于人工干预和审核。
* **技术可行性:** 文献信息提取、文本翻译和摘要生成是当前 AI 技术的成熟应用领域，Gemini API 具备实现这些功能的潜力。
* **人工审核保留:** 关键的价值判断环节 (亮点提炼、推荐理由生成) 仍保留人工审核，确保内容质量和符合社区定位。
* **逐步自动化:** 从简单的翻译、提取开始，逐步深入到更复杂的理解和生成任务，降低项目风险。
* **与 GIS/ABM 的关联:** 工作流从检索 ABM+GIS 文献开始，内容提炼也聚焦于 ABM+GIS 的融合方式和价值，与社区主题高度契合。Mesa 作为推广工具，其相关性可以在推荐理由中明确体现。

## 8. 如何参与 (How to Contribute - Optional)

如果您对本项目感兴趣，并希望参与其中，欢迎联系我们！您可以贡献：

* **文献筛选:** 协助阅读摘要进行初步筛选。
* **内容审核:** 协助校对和审核 AI 生成的亮点和推荐理由。
* **自动化开发:** 贡献 Python 编程能力，实现工作流自动化。
* **工具研究:** 探索其他有用的工具和方法。
* **提供建议:** 对工作流和内容形式提出宝贵意见。

[请在此处插入社区联系方式，例如微信号/邮箱/社群链接]

---

希望这份 README 文件能清晰地阐述项目规划，并为您提供一个坚实的基础来推进后续的自动化尝试。请您查阅！

## 10. 新用户快速上手指南（推荐使用 Conda 环境一键部署）

### 1. 安装 Conda（推荐 Miniconda 或 Anaconda）

- [Miniconda 下载地址](https://docs.conda.io/en/latest/miniconda.html)
- 安装完成后，确保 `conda` 命令可用。

### 2. 一键创建并激活项目环境

```bash
conda env create -f environment.yml
conda activate abmind-paper-bot
```

> 如无 `environment.yml`，可用如下命令手动创建：

```bash
conda create -n abmind-paper-bot python=3.10
conda activate abmind-paper-bot
pip install -r requirements.txt
```

### 3. 配置 Gemini API Key

- 在项目根目录下新建或编辑 `config_gemini.yaml`，内容如下：
  ```yaml
  gemini_api_key: "你的Gemini API Key"
  ```

### 4. 一键启动程序

```bash
streamlit run src/ui/streamlit_app.py
```

- 启动后访问命令行提示的网址（通常为 http://localhost:8501）。

### 5. 使用说明

- 按界面提示上传你的CSV文献文件，选择排序方式，即可自动生成标准Markdown内容。
- 输出内容可直接用于公众号、社群等平台。

### 6. 常见问题

- 如遇依赖冲突或启动异常，建议先删除旧环境再重试：
  ```bash
  conda deactivate
  conda remove -n abmind-paper-bot --all
  ```
- 如需 GPU/加速支持，请根据实际环境调整 `environment.yml`。

---
