import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.paper_recommender import PaperRecommender

st.set_page_config(page_title="ABMind 论文速递自动化工具", layout="wide")
st.title("ABMind 论文速递自动化工具")

st.markdown("""
本工具用于自动化筛选、整理和生成 ABM+GIS 领域的高质量论文推文。
""")

recommender = PaperRecommender()

uploaded_file = st.file_uploader("上传WOS导出的CSV文件", type=["csv"])

if uploaded_file:
    df = recommender.import_csv(uploaded_file)
    df_clean = recommender.clean_data(df)
    st.success(f"共导入 {len(df_clean)} 篇论文。")
    st.dataframe(df_clean.head(10))

    sort_option = st.selectbox("排序方式", ["最新优先", "最早优先"], index=0)
    n = st.number_input("本期推送论文数", min_value=1, max_value=50, value=20)
    if st.button("筛选并生成推文"):
        ascending = True if sort_option == "最早优先" else False
        df_selected = recommender.filter_and_sort(df_clean, n=n, ascending=ascending)
        papers = df_selected.to_dict(orient="records")
        progress_bar = st.progress(0, text="正在调用大模型自动整理论文内容，请耐心等待...")
        error_box = st.empty()
        def show_progress(current, total):
            progress_bar.progress(current / total, text=f"正在处理第{current}/{total}篇论文...")
        try:
            md = recommender.generate_markdown(papers, progress_callback=show_progress)
            progress_bar.empty()
            st.success("全部处理完成！")
            st.markdown("---\n#### 推文 Markdown 预览\n---")
            st.code(md, language="markdown")
            st.download_button("下载推文 Markdown", md, file_name="abmind_paper_digest.md")
        except Exception as e:
            progress_bar.empty()
            error_box.error(f"处理过程中发生错误: {e}")
