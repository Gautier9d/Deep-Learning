import streamlit as st
import pandas as pd
import os

def render_glossary():
    st.title("Glossary of Dogwhistles")
    st.markdown("We used the following dog whistles in our analysis.")
    st.markdown("For a more comprehensive list, see the [Allen AI Dogwhistles Glossary](https://dogwhistles.allen.ai/glossary).")
    
    # Read the CSV file
    csv_path = "dw_list.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)[["dog_whistle", "definition"]]
            
            # Apply custom styling
            st.markdown("""
                <style>
                    .stDataFrame {
                        font-size: 16px;
                    }
                    .stDataFrame thead tr th {
                        background-color: #f0f2f6;
                        font-weight: bold;
                    }
                    .stDataFrame tbody tr:nth-of-type(odd) {
                        background-color: #f8f9fa;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            # Display the dataframe with improved formatting
            st.dataframe(
                df.style.set_properties(**{
                    'text-align': 'left',
                    'white-space': 'pre-wrap'
                }),
                use_container_width=True,
                height=800,
            )
        except Exception as e:
            st.error(f"Error reading the CSV file: {e}")
    else:
        st.warning(f"Dogwhistle list file not found at {csv_path}")
