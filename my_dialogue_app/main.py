import os
import time
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from gptcache import cache, Config
from gptcache.adapter import openai as gptcache_openai
from gptcache.embedding import OpenAI
from gptcache.manager import get_data_manager, CacheBase, VectorBase
from gptcache.similarity_evaluation.distance import SearchDistanceEvaluation
from starlette.middleware.sessions import SessionMiddleware

# ✅ Init FastAPI + Jinja2
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(SessionMiddleware, secret_key="!secret123")

# ✅ Load OpenAI key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY not set!")

# ✅ Setup GPTCache
embedding_model = OpenAI(api_key=OPENAI_API_KEY)
data_manager = get_data_manager(
    CacheBase("sqlite", db_path="sqlite.db"),
    VectorBase("faiss", dimension=embedding_model.dimension, index_path="faiss.index")
)
cache.init(
    embedding_func=embedding_model.to_embeddings,
    data_manager=data_manager,
    similarity_evaluation=SearchDistanceEvaluation(),
    config=Config(similarity_threshold=0.95),
)

# ✅ HTML Home
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if "history" not in request.session:
        request.session["history"] = []
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "history": request.session["history"]}
    )


# ✅ Chat endpoint
@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, user_message: str = Form(...)):
    start = time.time()

    question_emb = embedding_model.to_embeddings([user_message])
    hits = cache.data_manager.search(question_emb, top_k=1)

    is_hit = False
    if hits:
        scalar_data = cache.data_manager.get_scalar_data(hits[0])
        if scalar_data:
            eval_query = {"question": user_message, "embedding": question_emb}
            eval_cache = {
                "question": scalar_data.question,
                "answer": scalar_data.answers[0].answer,
                "embedding": scalar_data.embedding_data,
                "search_result": hits[0]
            }
            similarity = cache.similarity_evaluation.evaluation(eval_query, eval_cache)
            min_rank, max_rank = cache.similarity_evaluation.range()
            threshold = (max_rank - min_rank) * cache.config.similarity_threshold
            if similarity >= threshold:
                is_hit = True

    response = gptcache_openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )
    answer = gptcache_openai.get_message_from_openai_answer(response)
    duration = time.time() - start

    # Append to session history
    history = request.session.get("history", [])
    history.append({
        "question": user_message,
        "answer": answer,
        "cached": is_hit,
        "time": f"{duration:.2f}s"
    })
    request.session["history"] = history

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "history": history}
    )
