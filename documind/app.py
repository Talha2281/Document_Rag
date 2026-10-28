import gradio as gr
from app.main import app as fastapi_app

# Gradio mount karta hai aapke existing FastAPI backend aur Frontend UI ko
demo = gr.mount_gradio_app(
    fastapi_app,
    gr.Blocks(),
    path="/gradio"
)

app = fastapi_app