# pyrag

基于 RAG（检索增强生成）的智能客服 Demo：上传 txt 知识文档到本地向量库，用户提问时先从知识库检索相关内容，再交给大模型组织成回答，并支持多轮对话记忆和流式输出。

## 技术栈

- **LangChain**：串联检索、提示词、模型、对话历史
- **Chroma**：本地向量数据库
- **通义千问（DashScope）**：`text-embedding-v4` 做向量化，`qwen3-max` 生成回答
- **Streamlit**：知识库上传页面和聊天页面

## 工作流程

```
上传 txt ──> 文本切分 ──> 向量化 ──> 存入 Chroma
                                        │
用户提问 ──> 向量检索相似片段 ─────────────┘
          ──> 拼接「参考资料 + 对话历史 + 问题」
          ──> qwen3-max 流式生成回答
```

## 目录结构

```
pyrag/
├── app_file_uploader.py   # 知识库上传页面
├── app_qa.py              # 智能客服聊天页面
├── knowledge_base.py      # 文本切分、MD5 去重、写入向量库
├── vector_store.py        # 向量库连接，提供检索器
├── rag.py                 # RAG 链：检索 + 提示词 + 模型 + 对话历史
├── config_data.py         # 配置项（模型名、切分参数、检索数量等）
├── data/                  # 示例知识文档
├── requirements.txt
└── .env.example
```

运行后会自动生成以下内容（已在 `.gitignore` 中忽略）：

- `chroma_db/`：向量库数据
- `md5.txt`：已入库内容的 MD5 记录，用于避免重复上传
- `chat_history/`：按会话保存的聊天记录

## 快速开始

### 1. 创建虚拟环境并安装依赖

需要 Python 3.10 及以上（开发时使用 3.13）。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

在 `.env` 中填入阿里云百炼的 API Key：

```
DASHSCOPE_API_KEY=你的key
```

### 3. 上传知识库

```bash
streamlit run app_file_uploader.py
```

在页面中逐个上传 `data/` 目录下的 txt 文件。相同内容重复上传会自动跳过。

### 4. 启动智能客服

```bash
streamlit run app_qa.py
```

然后就可以提问了，例如：「我身高 180，穿什么尺码？」

## 配置说明

在 `config_data.py` 中修改：

| 配置项 | 说明 |
|---|---|
| `embedding_model_name` | 向量化模型 |
| `chat_model_name` | 对话模型 |
| `chunk_size` / `chunk_overlap` | 文本切分长度和重叠字数 |
| `max_split_char_number` | 超过这个长度的文档才会切分 |
| `similarity_threshold` | 每次检索返回的片段数量 |
| `session_config` | 会话 ID，决定聊天记录保存在哪个文件 |

## 注意事项

- `chroma_db/` 和 `md5.txt` 要一起删除或一起保留。只删向量库、保留 `md5.txt`，会导致重新上传时被误判为"已存在"而跳过。
- 更换 `embedding_model_name` 后需要清空 `chroma_db/` 和 `md5.txt` 重新上传，不同模型生成的向量不能混用。
- 目前所有用户共用同一个会话 ID（`user_01`），聊天记录会互相叠加。
