import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import re
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Move the generate_headline function outside the class for proper caching
@st.cache_data(show_spinner=False)
def cached_generate_headline(article, _model, _tokenizer, _device):
    try:
        # Input validation
        if not article or len(article.strip()) < 10:
            raise ValueError("Article text is too short")
            
        # Preprocess
        article = re.sub(r'\s+', ' ', str(article))
        article = re.sub(r'[^\w\s.,!?]', '', article)
        article = article.strip()
        
        # Tokenize
        inputs = _tokenizer(
            article,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(_device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = _model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=64,
                num_beams=5,
                length_penalty=1.5,
                early_stopping=True
            )
        generation_time = time.time() - start_time
        logger.info(f"Headline generated in {generation_time:.2f} seconds")

        # Decode
        headline = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        return headline

    except Exception as e:
        logger.error(f"Error generating headline: {str(e)}")
        raise

class HeadlineGenerator:
    def __init__(self, model_name="Lord-Connoisseur/headline-generator"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            # Load model and tokenizer from Hugging Face Hub
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully from Hugging Face Hub")
            
        except Exception as e:
            logger.error(f"Error loading model from Hugging Face Hub: {str(e)}")
            raise

    def generate_headline(self, article):
        return cached_generate_headline(article, self.model, self.tokenizer, self.device)

def main():
    st.set_page_config(
        page_title="News Headline Generator",
        page_icon="📰",
        layout="wide"
    )

    st.title("📰 News Headline Generator")
    
    # Initialize model
    @st.cache_resource(show_spinner=False)
    def load_model():
        with st.spinner("Loading model from Hugging Face Hub... (this may take a minute)"):
            return HeadlineGenerator()

    try:
        model = load_model()
        st.success("✅ Model loaded successfully! Ready to generate headlines.")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Please make sure the model is correctly uploaded to Hugging Face Hub")
        st.stop()

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses BART to generate headlines for news articles.
        
        **Tips for best results:**
        - Provide clear, well-formatted text
        - Ideal length: 100-500 words
        - Include key information early
        
        Model powered by Hugging Face 🤗
        """)

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        article_text = st.text_area(
            "Enter news article text:",
            height=300,
            placeholder="Paste your article text here..."
        )

    with col2:
        st.markdown("### Settings")
        max_length = st.slider("Maximum headline length", 10, 100, 64)
        
    if st.button("Generate Headline", type="primary"):
        if article_text.strip():
            try:
                with st.spinner("🤖 Generating headline..."):
                    headline = model.generate_headline(article_text)
                
                st.success("✨ Headline generated!")
                st.subheader("Generated Headline:")
                st.markdown(f"### {headline}")
                
                # Additional metrics
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Article Length", f"{len(article_text.split())} words")
                with col2:
                    st.metric("Headline Length", f"{len(headline.split())} words")
                
            except ValueError as ve:
                st.warning(f"⚠️ {str(ve)}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Generation error: {str(e)}")
        else:
            st.warning("⚠️ Please enter some article text.")

if __name__ == "__main__":
    main()

