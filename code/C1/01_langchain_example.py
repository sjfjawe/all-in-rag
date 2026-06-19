import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知："抱歉，我无法根据提供的上下文找到相关信息来回答此问题。"

上下文:
{context}

问题: {question}

回答:""")

# 配置大语言模型 - 使用 Hugging Face Inference API
print("正在加载 DeepSeek-R1 模型 (via Hugging Face)...")

# 方案1: 使用 DeepSeek-R1 (推荐，强大的推理模型)
llm_endpoint = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_KEY_RAG_JUN26"),
    temperature=0.7,
    max_new_tokens=4096,
    timeout=120,
)

# 包装为 Chat 模型以便更好地处理对话
llm = ChatHuggingFace(llm=llm_endpoint)

# 备选方案 (如果 DeepSeek-R1 太慢或不可用):
# 方案2: Qwen (快速且支持中文)
# llm_endpoint = HuggingFaceEndpoint(
#     repo_id="Qwen/Qwen2.5-7B-Instruct-1M",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_KEY_RAG_JUN26"),
#     temperature=0.7,
#     max_new_tokens=4096,
# )
# llm = ChatHuggingFace(llm=llm_endpoint)

# 方案3: GLM (针对中文优化)
# llm_endpoint = HuggingFaceEndpoint(
#     repo_id="zai-org/GLM-4.5",
#     huggingfacehub_api_token=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGINGFACE_KEY_RAG_JUN26"),
#     temperature=0.7,
#     max_new_tokens=4096,
# )
# llm = ChatHuggingFace(llm=llm_endpoint)

print("模型加载完成！")

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
print(f"正在检索相关文档: {question}")
retrieved_docs = vectorstore.similarity_search(question, k=3)
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

print("正在生成答案...")
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print("\n" + "="*50)
print("回答:")
print("="*50)
print(answer.content if hasattr(answer, 'content') else answer)
