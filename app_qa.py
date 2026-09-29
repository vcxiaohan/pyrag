import streamlit as st
from rag import RagService
import config_data as config

st.title("智能客服")
st.divider()

user_input = st.chat_input("请输入")

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "请问有什么需要？"}]

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("human").write(user_input)
    with st.spinner("思考中..."):
        res = st.session_state["rag"].chain.stream({"input": user_input}, config.session_config)
        txt = st.chat_message("assistant").write_stream(res)
    st.session_state["messages"].append({"role": "assistant", "content": txt})
