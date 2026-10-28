import gradio as gr

from app.damage_pipeline import analyze


def _find_submit_button(demo):
    # gr.Interface creates both a Submit and a Clear button; pick Submit.
    buttons = [b for b in demo.blocks.values() if type(b) is gr.Button]
    for b in buttons:
        if str(getattr(b, "value", "")).strip().lower() == "submit":
            return b
    return buttons[0] if len(buttons) == 1 else None


def create_interface():
    image = gr.Image(type="filepath", label="Upload car image")

    demo = gr.Interface(
        fn=analyze,
        inputs=image,
        outputs=[
            gr.Image(type="pil", label="segmented_image"),
            gr.Textbox(label="Classification"),
            gr.Textbox(label="Detected_damage"),
            gr.Markdown(label="Damage summary", show_label=True, container=True),
        ],
        title="Car Damage Detection",
        flagging_mode="never",
    )

    submit = _find_submit_button(demo)
    if submit is not None:
        # Start disabled, and toggle on upload / clear.
        submit.interactive = False
        with demo:
            image.change(
                fn=lambda img: gr.update(interactive=img is not None),
                inputs=image,
                outputs=submit,
            )

    return demo
