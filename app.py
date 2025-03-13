from sanic import Sanic
from sanic.response import json
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from argparse import ArgumentParser

import os
import sglang as sgl
import chromadb
import PyPDF2
import asyncio


def parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="recoilme/recoilme-gemma-2-9B-v0.4"
    )
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--single_process", action="store_true", default=True)
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="sentence-transformers/static-similarity-mrl-multilingual-v1",
    )
    parser.add_argument("--n_results", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--stop_token_ids", type=int, nargs="+", default=[1])
    parser.add_argument("--stop", type=str, nargs="+", default=["</ASSISTANT>"])
    parser.add_argument("--renew_db", action="store_true", default=False)
    parser.add_argument("--max_chunk_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    return parser.parse_args()


engine = None

# Create an instance of the Sanic app
app = Sanic("rag-server")

# Create queue
request_queue = asyncio.Queue()


async def worker():
    while True:
        data, future = await request_queue.get()
        try:
            result = await get_academic_info(data)
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            request_queue.task_done()


# Define post method
# @app.route("/rag_qa", methods=["POST"])
async def get_academic_info(request):
    question = request["question"]
    query_embedding = app.ctx.retriever.encode([question]).tolist()
    if not question:
        return json({"error": "Question is required"}, status=400)

    # get the most similar document
    context = app.ctx.collection.query(
        query_embedding, n_results=config.n_results, include=["documents", "metadatas"]
    )
    context_prompt = ""
    for i, (doc, meta) in enumerate(
        zip(context["documents"][0], context["metadatas"][0])
    ):
        context_prompt += f"Context {i+1}:\nTitle(출처): {meta['file_name']}\n{doc}\n\n"

    stream = request["stream"]

    prompt = f"""<SYSTEM>
당신은 LLM 연구자를 위한 Assistant입니다.
아래 문서 내용을 참고하여 질문에 대해 답변을 제공하세요.
반드시 문서 내용을 참고하고, 출처를 표기해주세요.

{context_prompt}

</SYSTEM>
<USER>
{question}
</USER>
<ASSISTANT>"""
    if not stream:
        # async_generate returns a dict
        result = await engine.async_generate(
            prompt=prompt,
            sampling_params={
                "temperature": config.temperature,
                "max_new_tokens": config.max_new_tokens,
                "stop_token_ids": config.stop_token_ids,
                "stop": config.stop,
            },
        )

        return json(result)
    else:
        # async_generate returns a dict
        result = await engine.async_generate(
            prompt=prompt,
            sampling_params={
                "temperature": config.temperature,
                "max_new_tokens": config.max_new_tokens,
                "stop_token_ids": config.stop_token_ids,
                "stop": config.stop,
            },
            stream=True,
        )
        # init the response
        response = await request.respond()
        # result is an async generator
        async for chunk in result:
            await response.send(chunk["text"])

        await response.eof()


@app.route("/serve", methods=["POST"])
async def serve(request):
    data = request.json  # 요청의 JSON 데이터 추출
    loop = asyncio.get_event_loop()
    # 워커가 처리한 결과를 기다리기 위한 Future 객체 생성
    future = loop.create_future()
    # 요청 데이터와 future를 큐에 넣습니다.
    await request_queue.put((data, future))
    # 워커가 future에 결과를 채워줄 때까지 대기합니다.
    result = await future
    return result


def extract_text_from_html(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator="\n")
    return text


def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, max_chunk_size=1024, overlap=128):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + max_chunk_size])
        chunks.append(chunk)
        i += max_chunk_size - overlap
    return chunks


def load_retriever(
    model_name="sentence-transformers/static-similarity-mrl-multilingual-v1",
):
    return SentenceTransformer(model_name, device="cpu")


def init_db_client(data_path="./data"):
    return chromadb.Client(
        Settings(
            is_persistent=True,
            persist_directory=data_path,
        )
    )


@app.listener("before_server_start")
async def setup_rag_resources(app, loop):
    print("Setting up RAG resources...", flush=True)
    app.ctx.retriever = load_retriever(config.retriever_model)
    app.ctx.db_client = init_db_client(config.data_path)
    try:
        if config.renew_db:
            raise Exception("Setting renew_db to True.")
        app.ctx.collection = app.ctx.db_client.get_collection("rag_collection")
        print("Collection loaded.", flush=True)
    except:
        try:
            app.ctx.db_client.delete_collection("rag_collection")
            print("Collection deleted.", flush=True)
        except:
            print("No collection to delete.", flush=True)
        app.ctx.collection = app.ctx.db_client.create_collection("rag_collection")

        print("Waiting for documents to be added to the collection...", flush=True)
        # get texts for all docs in the files
        documents = []
        files = []
        for file in os.listdir(config.data_path):
            if file.endswith(".html"):
                text = extract_text_from_html(os.path.join(config.data_path, file))
                chunks = chunk_text(text, config.max_chunk_size, config.overlap)
                for chunk in chunks:
                    documents.append(chunk)
                    files.append(file)
            elif file.endswith(".pdf"):
                text = extract_text_from_pdf(os.path.join(config.data_path, file))
                chunks = chunk_text(text, config.max_chunk_size, config.overlap)
                for chunk in chunks:
                    documents.append(chunk)
                    files.append(file)
        embeddings = get_embeddings(documents, app.ctx.retriever)

        # add documents embeddins to the collection
        for i, (doc, embed, fname) in enumerate(zip(documents, embeddings, files)):
            app.ctx.collection.add(
                ids=[f"doc_{i}"],
                documents=[doc],
                embeddings=[embed],
                metadatas=[{"file_name": fname}],
            )
        print("Documents added to collection.", flush=True)
    print("RAG resources setup complete.", flush=True)
    print("Starting worker...", flush=True)
    loop.create_task(worker())


def get_embeddings(docs, retriever):
    embeddings = retriever.encode(docs)
    return embeddings.tolist()


def run_server():
    global engine, data_path, config
    config = parse()
    print("Loading engine...", flush=True)
    engine = sgl.Engine(model_path=config.model_path)
    data_path = config.data_path
    print(f"Starting server on {config.host}:{config.port}...", flush=True)
    app.run(host=config.host, port=config.port, single_process=config.single_process)


if __name__ == "__main__":
    run_server()
