import streamlit as st
import torch
from utils import load_transformer_rationale
from deep_dog.data.ss_utils import create_rationale_mask
import sys


@st.cache_resource
def load_model(model_type):
    """Load the model and tokenizer only once and cache them"""
    return load_transformer_rationale(model_type=model_type)


def get_prediction(sentence, model_type):
    """Get prediction for a given sentence and model"""
    model, tokenizer = load_model(model_type)
    
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
    
    return highlighted_text


# Initialize states if not present
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0
if 'predictions' not in st.session_state:
    st.session_state.predictions = {
        'sentence': None,
        'BERT': {'output': None},
        'BERT+LoRA': {'output': None}
    }

# Create four tabs
tab1, tab2, tab3, tab4 = st.tabs(["DeepDog", "Glossary", "Poster", "Model Demo"])

with tab1:
    from readme_streamlit import render_readme
    render_readme()

with tab2:
    from glossary_streamlit import render_glossary
    render_glossary()

with tab3:
    st.title("Poster")
    st.write(
        """
        <div style="position: relative; width: 100%; height: 0; padding-top: 141.5825%;
            padding-bottom: 0; box-shadow: 0 2px 8px 0 rgba(63,69,81,0.16); margin-top: 1.6em; margin-bottom: 0.9em; overflow: hidden;
            border-radius: 8px; will-change: transform;">
            <iframe loading="lazy" style="position: absolute; width: 100%; height: 100%; top: 0; left: 0; border: none; padding: 0;margin: 0;"
                src="https://www.canva.com/design/DAGoSCg6w0A/1QbWdrKJwfPmQ9OMh5tvSw/view?embed" allowfullscreen="allowfullscreen" allow="fullscreen">
            </iframe>
        </div>
        """,
        unsafe_allow_html=True
    )

with tab4:
    st.title("BERT Dog Whistle Demo")
    
    # Model selection
    model_type = st.radio(
        "Select Model",
        ["BERT", "BERT+LoRA"],
        help="BERT is the base model. BERT+LoRA uses parameter-efficient fine-tuning for improved performance."
    )
    
    sentence = st.text_input("Enter a sentence:", "This is a secret dog whistle message.")

    if st.button("Predict"):
        with st.spinner(f"Running prediction with {model_type}..."):
            # Get the new prediction
            new_output = get_prediction(sentence, model_type)
            
            # Reset predictions if sentence changed
            if sentence != st.session_state.predictions['sentence']:
                st.session_state.predictions = {
                    'sentence': sentence,
                    'BERT': {'output': None},
                    'BERT+LoRA': {'output': None}
                }
            
            # Store the new prediction
            st.session_state.predictions['sentence'] = sentence
            st.session_state.predictions[model_type] = {
                'output': new_output
            }
            
            # Show the predictions
            st.markdown("### Model Output (Dog whistles highlighted in red):")
            
            # If we have predictions from both models for the same sentence
            if (st.session_state.predictions['BERT']['output'] and 
                st.session_state.predictions['BERT+LoRA']['output']):
                col1, col2 = st.columns(2)
                
                # BERT Column
                with col1:
                    st.markdown("**BERT Output:**")
                    st.markdown(f"<div style='font-size: 1.2em'>{st.session_state.predictions['BERT']['output']}</div>", 
                              unsafe_allow_html=True)
                
                # BERT+LoRA Column
                with col2:
                    st.markdown("**BERT+LoRA Output:**")
                    st.markdown(f"<div style='font-size: 1.2em'>{st.session_state.predictions['BERT+LoRA']['output']}</div>", 
                              unsafe_allow_html=True)
            
            else:
                # Show only current prediction
                st.markdown(f"**{model_type} Output:**")
                st.markdown(f"<div style='font-size: 1.2em'>{new_output}</div>", 
                          unsafe_allow_html=True)
                
                # Add guidance message
                other_model = "BERT+LoRA" if model_type == "BERT" else "BERT"
                st.info(f"💡 Try selecting {other_model} and clicking Predict again to see a side-by-side comparison of both models' outputs for this sentence.")
