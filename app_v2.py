import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# ─── 1. PAGE CONFIGURATION ───
st.set_page_config(page_title="Class Social QA", page_icon="📚", layout="centered")
st.title("📚 सामाजिक अध्ययन प्रश्नोत्तर (Class 5-10)")
st.caption("Tap your question below. The system retrieves the exact answer from the curriculum database.")

# ─── 2. LOAD MODELS & DB (Cached so it loads only once) ───
@st.cache_resource(show_spinner="Loading Database and Model...")
def load_resources():
    # Update these paths/names to match your setup
    client = chromadb.PersistentClient(path="./db")
    col = client.get_collection("social_e5_large")
    embedder = SentenceTransformer("intfloat/multilingual-e5-large")
    return col, embedder

col, embedder = load_resources()

# ─── 3. SIDEBAR SETTINGS ───
with st.sidebar:
    st.header("⚙️ Settings")
    # Allow tweaking the strictness on the fly
    threshold = st.slider("Distance Threshold (Strictness)", 
                          min_value=0.05, max_value=0.50, value=0.12, step=0.01,
                          help="Lower = exact matches only. Higher = allows looser matches.")
    
    st.info(f"📊 Total Q&A in Database: **{col.count()}**")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ─── 4. CHAT HISTORY STATE ───
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show retrieved source if it was an assistant message
        if msg["role"] == "assistant" and "source" in msg:
            with st.expander("🔍 View Source & Score"):
                st.write(f"**Distance:** {msg['source']['distance']:.3f}")
                st.write(f"**Unit:** {msg['source']['unit']}")
                st.write(f"**Lesson:** {msg['source']['lesson']}")
                st.write(f"**Raw Document:** {msg['source']['document']}")

# ─── 5. CHAT INPUT & RAG LOGIC ───
if prompt := st.chat_input("आफ्नो प्रश्न यहाँ लेख्नुहोस्... (Ask your question)"):
    
    # A. Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # B. Process RAG Query
    with st.chat_message("assistant"):
        with st.spinner("खोज्दै... (Searching...)"):
            
            # 1. Embed Query
            q_vec = embedder.encode([f"query: {prompt}"], normalize_embeddings=True)[0].tolist()
            
            # 2. Search DB
            results = col.query(
                query_embeddings=[q_vec], 
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )

            answer = ""
            source_info = None

            if not results["ids"][0]:
                answer = "माफ गर्नुहोस्, यो प्रश्नको उत्तर सामग्रीमा छैन।"
            else:
                distance = results["distances"][0][0]
                metadata = results["metadatas"][0][0]
                document_text = results["documents"][0][0]

                # 3. Check Threshold
                if distance > threshold:
                    answer = f"माफ गर्नुहोस्, यो प्रश्नको उत्तर सामग्रीमा छैन। (Distance: {distance:.3f})"
                else:
                    # 4. Extract Exact Answer
                    answer = metadata.get("answer") or metadata.get("response") or "Answer format error."
                    source_info = {
                        "distance": distance,
                        "unit": metadata.get("unit", "N/A"),
                        "lesson": metadata.get("lesson", "N/A"),
                        "document": document_text
                    }

            # C. Display Answer
            st.markdown(answer)
            
            # D. Display Source Details
            if source_info:
                with st.expander("🔍 View Source & Score"):
                    st.write(f"**Distance:** {source_info['distance']:.3f}")
                    st.write(f"**Unit:** {source_info['unit']}")
                    st.write(f"**Lesson:** {source_info['lesson']}")
                    st.write(f"**Raw Document:** {source_info['document']}")

            # E. Save to history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "source": source_info
            })
