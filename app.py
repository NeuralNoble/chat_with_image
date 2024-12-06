import streamlit as st
from PIL import Image
from io import BytesIO
from transformers import ViltProcessor, ViltForQuestionAnswering

st.set_page_config(layout="wide", page_title="VQA")

cache_dir = "./model_cache"


# vilt code
@st.cache_resource  # Add caching to prevent reloading model
def load_model():
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa",cache_dir=cache_dir,local_files_only=False)
    model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa",
        cache_dir=cache_dir,local_files_only=False
    )
    return processor, model


processor, model = load_model()


def get_answer(image, text):
    try:
        # Load and process the image
        if isinstance(image, bytes):
            img = Image.open(BytesIO(image)).convert('RGB')
        else:
            img = image.convert('RGB')

        # prepare input
        encoding = processor(img, text, return_tensors="pt")

        # forward pass
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        # Fix the typo in config
        answer = model.config.id2label[idx]

        return answer

    except Exception as e:
        return f"Error: {str(e)}"


st.title("Chat With Image")
st.write("Upload an Image and ask a question about the Image")

col1, col2 = st.columns(2)

# Image Upload
with col1:
    uploaded_file = st.file_uploader("Upload a Image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, use_column_width=True)

with col2:
    question = st.text_input("What is your question?")

    if uploaded_file is not None and question:  # Better condition checking
        if st.button("Ask Question"):
            image = Image.open(uploaded_file)
            answer = get_answer(image, question)  # Pass image directly
            st.info(f"Your Question: {question}")
            st.success(answer)