import streamlit as st
import torch
from codecarbon import EmissionsTracker
from utils import load_transformer_rationale
from deep_dog.data.ss_utils import create_rationale_mask
from deep_dog.utils import get_car_miles, get_household_fraction
import sys


# Initialize tab state if not present
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Create three tabs
tab1, tab2, tab3 = st.tabs(["DeepDog", "Glossary", "Model Demo"])

with tab1:
    from readme_streamlit import render_readme
    render_readme()

with tab2:
    from glossary_streamlit import render_glossary
    render_glossary()

with tab3:
    st.title("BERT Dog Whistle Demo")
    sentence = st.text_input("Enter a sentence:", "This is a secret dog whistle message.")

    if st.button("Predict"):
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
            rationale_scores = model(batch)[0]
            rationale_mask = (rationale_scores > 0.5).int().tolist()
        
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        filtered = [(tok, mask) for tok, mask in zip(tokens, rationale_mask)
                   if not (tok.startswith('[') and tok.endswith(']'))]
        
        # Reconstruct the sentence with rationale tokens highlighted in red
        highlighted_text = ""
        for tok, mask in filtered:
            word = tok[2:] if tok.startswith("##") else (" " + tok if highlighted_text else tok)
            if mask:
                highlighted_text += f'<span style="color:red;font-weight:bold">{word}</span>'
            else:
                highlighted_text += word
        
        st.markdown("### Model Output (Dog whistle highlighted):")
        st.markdown(f"<div style='font-size: 1.2em'>{highlighted_text}</div>", unsafe_allow_html=True)
