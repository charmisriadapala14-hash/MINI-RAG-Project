import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import numpy as np
import os

st.set_page_config(page_title="Mini RAG Application", page_icon="🤖")

st.title("🤖 Mini RAG Application")
st.write("Ask questions based on your documents.")

api_key = st.sidebar.text_input("Enter OpenRouter API Key", type="password")

document = st.text_area(
"Enter your document",
placeholder="Paste your document content here..."
)

question = st.text_input("Ask a question")

@st.cache_resource
def load_model():
return SentenceTransformer("all-MiniLM-L6-v2")

if st.button("Get Answer"):
if not api_key or not document or not question:
st.warning("Please enter API key, document and question.")
else:
with st.spinner("Finding relevant information..."):
model = load_model()

        chunks = [
            document[i:i+500]
            for i in range(0, len(document), 500)
        ]

        embeddings = model.encode(chunks)
        query_embedding = model.encode([question])

        similarities = cosine_similarity(
            query_embedding, embeddings
        )[0]

        best_index = int(np.argmax(similarities))
        context = chunks[best_index]

        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer using only the provided context. If the answer is not available, say so."
                },
                {
                    "role": "user",
                    "content": f"Context: {context}\n\nQuestion: {question}"
                }
            ]
        )

        st.subheader("Answer")
        st.write(response.choices[0].message.content)

        with st.expander("Retrieved Context"):
            st.write(context)
