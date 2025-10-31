from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from rag_chroma import chain as rag_chroma_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

add_routes(app, rag_chroma_chain, path="/rag-chroma")

# If you want to expose additional chains, call add_routes with the chain module.
# The previous call used NotImplemented which causes a runtime error, so it's removed.
# Example: add_routes(app, another_chain, path="/other-chain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
