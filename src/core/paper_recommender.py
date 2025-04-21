"""
核心论文推荐与推文生成逻辑
"""

import pandas as pd
from typing import List, Dict, Any
import yaml
from .ai_provider import GeminiProvider

class PaperRecommender:
    def __init__(self, db_path: str = None, gemini_config: str = None):
        import os
        self.db_path = db_path
        # 自动查找API Key
        config_paths = []
        if gemini_config:
            config_paths.append(gemini_config)
        # 项目根目录
        config_paths.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config_gemini.yaml')))
        config_paths.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config_gemini.yaml')))
        api_key = None
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        config = yaml.safe_load(f)
                        api_key = config.get("gemini_api_key")
                        if api_key:
                            break
                except Exception:
                    continue
        if not api_key:
            print("[ERROR] 未能加载Gemini API Key，请确保config_gemini.yaml在项目根目录且内容正确！")
        self.gemini = GeminiProvider(api_key) if api_key else None

    def import_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # 自动识别常见的标题和DOI字段名
        title_candidates = ["Title", "Article Title", "TI", "标题"]
        doi_candidates = ["DOI", "DOI Link", "doi"]
        title_col = next((col for col in title_candidates if col in df.columns), None)
        doi_col = next((col for col in doi_candidates if col in df.columns), None)
        subset = []
        if title_col:
            subset.append(title_col)
        if doi_col:
            subset.append(doi_col)
        if subset:
            df = df.drop_duplicates(subset=subset, keep="first")
        else:
            df = df.drop_duplicates(keep="first")
        return df

    def filter_and_sort(self, df: pd.DataFrame, n: int = 20, ascending: bool = False) -> pd.DataFrame:
        # 按年份和日期排序，ascending=True为最早优先，False为最新优先
        if "Publication Year" in df.columns:
            df = df.sort_values(by=["Publication Year", "Publication Date"], ascending=[ascending, ascending])
        return df.head(n)

    def generate_markdown(self, papers: List[Dict[str, Any]], progress_callback=None) -> str:
        import logging
        md = ""
        total = len(papers)
        for idx, paper in enumerate(papers, 1):
            abstract = paper.get('Abstract', '') or paper.get('摘要', '')
            authors = paper.get('Authors', '') or paper.get('Author', '') or paper.get('作者', '')
            journal = paper.get('Journal', '') or paper.get('Source title', '') or paper.get('期刊', '')
            year = paper.get('Publication Year', '') or paper.get('Year', '') or paper.get('发表年份', '')
            doi = paper.get('DOI', '') or paper.get('DOI Link', '') or paper.get('doi', '')
            title = paper.get('Title', '') or paper.get('Article Title', '') or paper.get('TI', '') or paper.get('标题', '')
            if not title.strip():
                zh_title = '[缺少英文标题，无法翻译]'
                keywords_en = keywords_zh = highlights = reasons = '[缺少英文标题，无法生成内容]'
                md += f"\n---\n\n**第 {idx} 篇**\n\n**【论文标题】**\n[缺少英文标题，无法翻译]\n\n**【作者】**\n{authors}\n\n**【期刊/来源】**\n{journal}, {year}\n{doi}\n\n**【摘要】**\n{abstract}\n\n**【关键词】**\n[缺少英文标题，无法生成内容]\n\n**【研究亮点】✨**\n[缺少英文标题，无法生成内容]\n\n**【推荐理由】👍**\n[缺少英文标题，无法生成内容]\n"
                if progress_callback:
                    progress_callback(idx, total)
                continue

            zh_title = zh_abstract = keywords_en = keywords_zh = highlights = reasons = ""
            try:
                if not self.gemini:
                    raise Exception("Gemini API 未初始化，请检查API Key和依赖。")
                # 1. 标题翻译（只输出标准学术风格翻译，不要任何解释说明、推荐、总结等内容）
                zh_title = self.gemini.generate_text(f"请将下列英文论文标题翻译成标准学术中文，仅输出翻译本身：\n{title}")
                # 2. 摘要翻译（只输出翻译本身）
                zh_abstract = self.gemini.generate_text(f"请将下列英文论文摘要翻译成标准学术中文，仅输出翻译本身：\n{abstract}")
                # 3. 关键词提取（英文关键词，逗号分隔，仅输出关键词本身）
                keywords_en = self.gemini.generate_text(f"请从下列英文论文标题和摘要中提取3-5个英文关键词，用逗号分隔，仅输出关键词本身：\n标题：{title}\n摘要：{abstract}")
                # 4. 关键词翻译（仅输出中文关键词本身）
                keywords_zh = self.gemini.generate_text(f"请将下列英文关键词翻译成标准学术中文，仅输出翻译本身：{keywords_en}")
                # 5. 研究亮点（直接输出1-3条中文亮点内容，每条独立一行，不要任何前缀、解释、说明、总结等）
                highlights = self.gemini.generate_text(f"请基于下列论文摘要，直接输出1-3条最重要的研究亮点，用中文分点，每条独立一行，仅输出亮点内容本身：\n{abstract}")
                # 6. 推荐理由（直接输出2-4条中文推荐理由，每条独立一行，不要任何前缀、解释、说明、总结等）
                reasons = self.gemini.generate_text(f"请基于下列论文内容，直接输出2-4条简洁有吸引力的推荐理由，用中文分点，每条独立一行，仅输出理由内容本身：\n标题：{title}\n摘要：{abstract}")
            except Exception as e:
                logging.error(f"Gemini调用失败: {e}")
                zh_title = zh_abstract = keywords_en = keywords_zh = highlights = reasons = f"[Gemini调用失败: {e}]"
            # 只输出一个DOI Link
            doi_link = paper.get('DOI Link', '') or (f'https://doi.org/{doi}' if doi and not str(doi).startswith('http') else doi)
            # 亮点与推荐理由直接输出，无前缀、无解释、无多版本
            md += f"\n---\n\n**第 {idx} 篇**\n\n**【论文标题】**\n{title}\n{zh_title}\n\n**【作者】**\n{authors}\n\n**【期刊/来源】**\n{journal}, {year}\n{doi_link}\n\n**【摘要】**\n{abstract}\n{zh_abstract}\n\n**【关键词】**\n{keywords_en}\n{keywords_zh}\n\n**【研究亮点】✨**\n"
            # 亮点多行为分点输出（如LLM返回多行）
            if highlights:
                for line in highlights.split('\n'):
                    if line.strip():
                        md += f"*   {line.strip()}\n"
            md += "\n**【推荐理由】👍**\n"
            if reasons:
                for line in reasons.split('\n'):
                    if line.strip():
                        md += f"*   {line.strip()}\n"

            if progress_callback:
                progress_callback(idx, total)
        return md
