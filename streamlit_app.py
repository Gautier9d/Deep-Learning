import streamlit as st
import torch
from codecarbon import EmissionsTracker
from utils import load_transformer_rationale
from deep_dog.data.ss_utils import create_rationale_mask
from deep_dog.utils import get_car_miles, get_household_fraction
import sys
st.title("BERT Dog Whistle Demo")
sentence = st.text_input("Enter a sentence:", "This is a secret dog whistle message.")

# Placeholder for output below the input box
output_placeholder = st.container()

# Use session state to manage button state
if 'predict_disabled' not in st.session_state:
    st.session_state['predict_disabled'] = False

def run_prediction():
    st.session_state['predict_disabled'] = True
    with st.spinner("Predicting..."):
        model, tokenizer = load_transformer_rationale()
        inputs = tokenizer(sentence,
                        max_length=256,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt")
        batch = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        with torch.no_grad():
            rationale_scores = model(batch)[0]  # [seq_len]
            rationale_mask = (rationale_scores > 0.5).int().tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        # Filter out special tokens (tokens that start with '[' and end with ']')
        filtered = [
            (tok, mask) for tok, mask in zip(tokens, rationale_mask)
            if not (tok.startswith('[') and tok.endswith(']'))
        ]
        # Use rationale mask to find dog whistle (first rationale token)
        dog_whistle = next((tok for tok, mask in filtered if mask), "None detected")
        # Reconstruct the sentence with rationale tokens highlighted in red
        highlighted_text = ""
        for tok, mask in filtered:
            # Handle subword tokens
            if tok.startswith("##"):
                word = tok[2:]
            else:
                word = " " + tok if highlighted_text else tok
            if mask:
                highlighted_text += f'<span style="color:red;font-weight:bold">{word}</span>'
            else:
                highlighted_text += word
        st.session_state['highlighted_text'] = highlighted_text
        st.session_state['dog_whistle'] = dog_whistle
    st.session_state['predict_disabled'] = False

st.button("Predict", on_click=run_prediction, disabled=st.session_state['predict_disabled'])

# Show output below the input box, if available
if 'highlighted_text' in st.session_state:
    output_placeholder.markdown("### Model Output Dog whistle highlighted:")
    output_placeholder.markdown(f"<div style='font-size: 1.2em'>{st.session_state['highlighted_text']}</div>", unsafe_allow_html=True)
