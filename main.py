import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# 加载环境变量
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# 加载文件夹里的所有文档
def load_documents(folder_path: str):
    loader = DirectoryLoader(
        folder_path,
        loader_cls=UnstructuredFileLoader,
        use_multithreading=True,
        recursive=True
    )
    return loader.load()

# 文本切分
def split_documents(documents):
    splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def load_embeddings():
    model_name = "../AgentRag/Camel_agent/models/multilingual-e5-large"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={
            "normalize_embeddings": True  
        }
    )
    return embeddings

# 向量化 + FAISS 知识库构建
def build_vectorstore(chunks, persist_path: str):
    embeddings = load_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(persist_path)
    return vectorstore

# 加载已持久化的 FAISS 向量库
def load_vectorstore(persist_path: str):
    #embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=256)
    embeddings = load_embeddings()
    return FAISS.load_local(
        persist_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def build_rag(query, vectorstore):
    query = f"query: {query}"
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrievers = retriever.invoke(query)
    return retrievers

# QA 系统构建
def build_qa(input,contexts,metadata):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=None
    )
    metadata = [i.split("/")[-1] for i in metadata]
    content = ""
    for index, context in enumerate(contexts):
        content += f"{index+1}. {context.page_content}"
        content += "\n"
    content = content.strip()

    messages = [
        (
            "system",
            "你是一个金融专家，请根据用户的问题和相关的知识库内容，给出专业的回答。",
        ),
        ("human", f"用户的问题是：{input},可能的相关的知识库内容是：{content}。必须告诉用户参考的源文件有这几个：{metadata}"),
    ]
    return llm, messages
def main(question):

    """print("加载并处理文档...")
    docs = load_documents(folder_path)
    chunks = split_documents(docs)
    new_chunks = []
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"
        new_chunks.append(chunk)

    print("构建向量库并持久化...")
    build_vectorstore(new_chunks, persist_path)"""
    print("初始化问答系统...")
    print(30*"=","开始问答")
    query = question.strip()
    print(30*"=","用户的问题是：",query)
    retrievers = build_rag(query, vectorstore)
    print(30*"=","检索到的相关内容是：",retrievers)
    metadata = [doc.metadata.get('source') for doc in retrievers]
    llm, messages = build_qa(input=query,contexts=retrievers,metadata=metadata)
    result = llm.invoke(messages)
    output = result.content
    print(30*"=","AI的回答是：",output)
    return output

if __name__ == "__main__":
    import gradio as gr
    folder_path = "bank/bank"
    persist_path = "faiss_index"
    print("加载向量库 + 构建 ReRank 检索器...")
    vectorstore = load_vectorstore(persist_path)
    print(30*"=","向量库加载完成")
    with gr.Blocks(title="Banking Knowledge Question and Answer Assistant") as demo:
        gr.Markdown("## 💰 Banking Knowledge Question and Answer Assistant")
        gr.Markdown("Enter your question about banking knowledge and I will generate professional answers for you based on the knowledge base.")

        with gr.Row():
            with gr.Column():
                user_input = gr.Textbox(label="User Question", placeholder="Please enter your question...", lines=5)
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear")

            with gr.Column():
                output = gr.Textbox(label="AI Answer", lines=10, interactive=False)

        submit_btn.click(fn=main, inputs=user_input, outputs=output)
        clear_btn.click(fn=lambda: ("", ""), inputs=[], outputs=[user_input, output])


    demo.launch(
        server_name="0.0.0.0",  # 允许所有IP访问
        server_port=7860,
        share=True
    )


