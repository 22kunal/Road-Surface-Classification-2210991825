import os
import gradio as gr
import numpy as np
import tensorflow as tf

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the trained CNN model
print("Loading Road Surface Classification Model...")
model = tf.keras.models.load_model('road_classification_model (1).keras')
print("Model loaded successfully!")

# Class names — alphabetical order (how TensorFlow reads folders)
class_names = ['Dry', 'Muddy', 'Wet']

# Description for each class
DESCRIPTIONS = {
    'Dry':   '✅ Dry Road — Normal traction. Safe driving conditions.',
    'Wet':   '⚠️ Wet Road — Reduced traction. Increase braking distance.',
    'Muddy': '🚨 Muddy Road — Low traction. Drive with extreme caution.',
}

# Prediction function
def predict(image):
    if image is None:
        return {}, "Please provide an image or use webcam."

    # Preprocess
    img = tf.image.resize(image, [128, 128])
    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)

    # Predict
    predictions = model.predict(img, verbose=0)
    probs = tf.nn.softmax(predictions[0]).numpy()

    # Build results
    results = {class_names[i]: float(probs[i]) for i in range(3)}
    top = class_names[np.argmax(probs)]
    confidence = float(np.max(probs)) * 100
    status = f"{DESCRIPTIONS[top]}\nConfidence: {confidence:.1f}%"

    return results, status

# Custom CSS
css = """
.gradio-container {
    background: linear-gradient(135deg, #0d0d0d, #1a1a2e) !important;
    font-family: 'Segoe UI', sans-serif !important;
    min-height: 100vh;
}
.gr-button-primary {
    background: #e53935 !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
}
.gr-button-primary:hover {
    background: #b71c1c !important;
}
label {
    color: #ffffff !important;
    font-weight: 600 !important;
}
footer { display: none !important; }
"""

# Build the interface
with gr.Blocks(title="Road Surface Classifier") as demo:

    gr.HTML("""
    <div style="text-align:center; padding:25px 20px 15px;
                border-bottom:3px solid #e53935; margin-bottom:20px;">
        <h1 style="color:white; font-size:2.2em; margin:0; font-weight:700;
                   letter-spacing:1px;">
            🛣️ Road Surface Classifier
        </h1>
        <p style="color:#e53935; margin:8px 0 4px; font-size:1.1em; font-weight:600;">
            CNN + KNN Hybrid Deep Learning Model
        </p>
        <p style="color:#cccccc; font-size:0.88em; margin:0;">
            Kunal Kumar &nbsp;|&nbsp; Roll No: 2210991825 &nbsp;|&nbsp;
            Chitkara University, Punjab
        </p>
    </div>

    <div style="background:rgba(229,57,53,0.15); border:1px solid #e53935;
                border-radius:10px; padding:12px 20px; text-align:center;
                color:#ffffff; margin-bottom:20px; font-size:0.92em;">
        <b style="color:#e53935;">How it works:</b>
        &nbsp; CNN extracts deep visual features from road image
        &nbsp;→&nbsp; KNN classifies based on those features
        &nbsp;→&nbsp; Detects: <b>Dry</b> / <b>Wet</b> / <b>Muddy</b>
    </div>
    """)

    with gr.Row():
        # Left — Input
        with gr.Column(scale=3):
            gr.Markdown(
                "<p style='color:white; font-size:1.1em; font-weight:600;'>📷 Input — Point webcam at road or upload image</p>"
            )
            img_input = gr.Image(
                sources=["webcam", "upload"],
                type="numpy",
                label="Webcam / Upload Road Image",
                height=320,
            )
            with gr.Row():
                btn = gr.Button(
                    "🔍 CLASSIFY ROAD SURFACE",
                    variant="primary",
                    size="lg"
                )
                clr = gr.Button("🔄 Clear", size="lg")

        # Right — Output
        with gr.Column(scale=2):
            gr.Markdown(
                "<p style='color:white; font-size:1.1em; font-weight:600;'>📊 Prediction Result</p>"
            )
            label_out = gr.Label(
                label="Road Surface Type",
                num_top_classes=3
            )
            status_out = gr.Textbox(
                label="Detection Status",
                lines=3,
                interactive=False
            )

            gr.HTML("""
            <div style="background:rgba(255,255,255,0.12);
                        border:1px solid rgba(255,255,255,0.25);
                        border-radius:10px; padding:15px;
                        color:#ffffff; font-size:0.9em;
                        line-height:1.9; margin-top:10px;">
                <p style="color:#e53935; font-weight:700; margin:0 0 8px;">
                    📊 Model Performance
                </p>
                <p style="color:#ffffff;">✅ Architecture: 4 Conv + 4 MaxPool + KNN</p>
                <p style="color:#ffffff;">✅ Paper Accuracy: <b style="color:#00e676;">98.31%</b></p>
                <p style="color:#ffffff;">✅ Beats: ResNet50, LSTM, SVM, Random Forest</p>
                <hr style="border-color:rgba(255,255,255,0.2); margin:10px 0;">
                <p style="color:#e53935; font-weight:700; margin:0 0 8px;">
                    🎯 Detects
                </p>
                <p style="color:#ffffff;">🟡 Dry Road — Safe conditions</p>
                <p style="color:#ffffff;">🔵 Wet Road — Reduced traction</p>
                <p style="color:#ffffff;">🟤 Muddy Road — Low traction</p>
                <hr style="border-color:rgba(255,255,255,0.2); margin:10px 0;">
                <p style="color:#cccccc; font-size:0.85em;">
                    💡 Point webcam at road or upload a road image.
                    Works best with clear, well-lit photos.
                </p>
            </div>
            """)

    # Wire buttons
    btn.click(
        fn=predict,
        inputs=[img_input],
        outputs=[label_out, status_out]
    )
    clr.click(
        fn=lambda: (None, {}, ""),
        outputs=[img_input, label_out, status_out]
    )
    img_input.change(
        fn=predict,
        inputs=[img_input],
        outputs=[label_out, status_out]
    )

demo.launch(css=css)
