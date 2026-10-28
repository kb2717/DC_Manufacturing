import os
from app.api.router import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        # 0.0.0.0 so the app is reachable from the network, not just localhost.
        server_name=os.getenv("APP_HOST", "127.0.0.1"),
        server_port=int(os.getenv("APP_PORT", "7860")),
        # share=True gives a temporary *.gradio.live URL that expires in 72h.
        share=os.getenv("APP_SHARE", "").lower() in ("1", "true", "yes"),
    )
