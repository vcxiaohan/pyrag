import os
import hashlib
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from vector_store import VectorStoreService

import config_data as config

load_dotenv()


def print_msgs(msgs):
    print("-" * 20 + "\n", msgs.to_string(), "\n" + "-" * 20)
    return msgs


def get_history(session_id):
    dir_path = "./chat_history"
    os.makedirs(dir_path, exist_ok=True)

    return FileChatMessageHistory(
        file_path=f"{dir_path}/{session_id}.json",
        encoding="utf-8",
        ensure_ascii=False
    )


class RagService(object):
    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "以我提供的已知参考资料为主，"
                       "简洁和专业的回答用户问题。参考资料:\n{context}。"),
            ("system", "并且我提供用户的对话历史记录，如下："),
            MessagesPlaceholder("history"),
            ("user", "请回答用户提问: {input}")
        ])

        # 新版 langchain-core 下 ChatTongyi 默认会被判定为关闭流式，必须显式开启
        self.chat_model = ChatTongyi(model=config.chat_model_name, streaming=True)

        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        # 拼接参考资料
        def format_docs(docs):
            str = "["
            for doc in docs:
                str += doc.page_content
            str += "]"
            return str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = {
                    "input": RunnablePassthrough(),
                    "context": RunnableLambda(format_for_retriever) | retriever | format_docs
                } | RunnableLambda(
            format_for_prompt_template) | self.prompt_template | print_msgs | self.chat_model | StrOutputParser()

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history"
        )

        return conversation_chain


if __name__ == "__main__":
    result = RagService().chain.invoke({"input": "我多高多重"}, config.session_config)
    print(result)
