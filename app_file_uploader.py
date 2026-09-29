import time

import streamlit as st
from knowledge_base import KnowledgeBaseService

st.title("客服机器人")

uploader_file = st.file_uploader(
    "请上传txt文件",
    type=["txt"],
    accept_multiple_files=False
)

if "service" not in st.session_state:
    st.session_state["service"] = KnowledgeBaseService()

if uploader_file is not None:
    # uploader_file.name
    text = uploader_file.getvalue().decode("utf-8")
    # st.write(text)

    with st.spinner("载入数据库中..."):
        result = st.session_state["service"].upload_by_str(text, uploader_file.name)
        time.sleep(2)
        st.write(result)
