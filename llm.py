import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import gtts
from gtts import gTTS
from io import BytesIO
import os

# Load the model, tokenizer, and feature extractor
model = VisionEncoderDecoderModel.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
feature_extractor = ViTImageProcessor.from_pretrained('nlpconnect/vit-gpt2-image-captioning')
tokenizer = AutoTokenizer.from_pretrained('nlpconnect/vit-gpt2-image-captioning')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {'max_length': max_length, 'num_beams': num_beams}

def predict_step(image):
    pixel_values = feature_extractor(images=[image], return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)

    attention_mask = torch.ones(pixel_values.shape[:-1], device=device)
    output_ids = model.generate(pixel_values, attention_mask=attention_mask, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0]
def text_to_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

if __name__=='__main__':
    # Streamlit app
    st.title('Image Captioning with Audio Output')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Button to trigger caption generation and audio output
        if st.button('Generate Caption and Audio'):
            st.write("Generating caption...")
            caption = predict_step(image)
            st.write("Caption:", caption)
            # Generate audio
            audio_buffer = text_to_audio(caption)
            st.audio(audio_buffer, format='audio/mp3')