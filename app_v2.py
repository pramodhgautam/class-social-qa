import os
import json
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

# ─── 1. PAGE CONFIGURATION ───
st.set_page_config(page_title="Class Social QA", page_icon="📚", layout="centered")

# Mobile Responsive CSS
st.markdown("""
<style>
    .block-container { max-width: 100% !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    @media (max-width: 768px) {
        .stChatMessage { padding: 0.75rem !important; }
        button, [role="button"] { min-height: 44px !important; font-size: 16px !important; }
        input, textarea { font-size: 16px !important; }
        .streamlit-expanderHeader { width: 100% !important; }
    }
    @media (min-width: 769px) {
        .block-container { max-width: 900px !important; margin: 0 auto; }
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 सामाजिक अध्ययन प्रश्नोत्तर (Class 5-10)")
st.caption("Tap your question below. The system retrieves the exact answer from the curriculum database.")

# ─── 2. LOAD MODELS & DB (Auto-rebuilds if missing) ───
@st.cache_resource(show_spinner="Loading Database and Model...")
def load_resources():
    # Load the embedder
    embedder = SentenceTransformer("intfloat/multilingual-e5-large")
    
    # 1. Try to load existing DB
    try:
        client = chromadb.PersistentClient(path="./db")
        col = client.get_collection("social_e5_large")
        if col.count() > 0:
            return col, embedder
    except Exception:
        pass # DB missing or corrupted -> Rebuild below

    # 2. Rebuild DB from JSONL (Fixes Streamlit Cloud missing collection error)
    print("⚠️ DB not found or corrupted. Rebuilding from JSONL...")
    
    # Ensure the JSONL file exists in your repo with this exact name
    if not os.path.exists("class_social_qa.jsonl"):
        st.error("Critical Error: class_social_qa.jsonl not found in repository!")
        st.stop()
        
    client = chromadb.PersistentClient(path="./db")
    col = client.create_collection(
        name="social_e5_large",
        metadata={"hnsw:space": "cosine"}
    )
    
    with open("class_social_qa.jsonl", "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
        
    docs, metas, ids, embed_inputs = [], [], [], []
    
    for i, r in enumerate(rows):
        doc = f"प्रश्न: {r['instruction']}\nउत्तर: {r['response']}"
        docs.append(doc)
        metas.append({
            "instruction": r["instruction"], 
            "answer": r["response"], 
            "unit": r.get("unit", ""), 
            "lesson": r.get("lesson", "")
        })
        ids.append(f"qa_{i}")
        # E5 requires "passage: " prefix for documents
        embed_inputs.append(f"passage: {doc}") 
        
    vectors = embedder.encode(embed_inputs, normalize_embeddings=True, batch_size=32)
    col.add(ids=ids, embeddings=vectors.tolist(), documents=docs, metadatas=metas)
    print(f"✅ DB successfully rebuilt with {len(ids)} documents.")
    
    return col, embedder

col, embedder = load_resources()

# ─── 3. SIDEBAR SETTINGS ───
with st.sidebar:
    st.header("⚙️ Settings")
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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "source" in msg:
            with st.expander("🔍 View Source & Score"):
                st.write(f"**Distance:** {msg['source']['distance']:.3f}")
                st.write(f"**Unit:** {msg['source']['unit']}")
                st.write(f"**Lesson:** {msg['source']['lesson']}")
                st.write(f"**Raw Document:** {msg['source']['document']}")

# ─── 5. SAMPLE QUESTIONS (Quick Prompts) ───
st.markdown("##### 💡 नमुना प्रश्नहरू (Sample Questions)")
sample_questions = [
    "समुदाय भनेको के हो?",
    "सामुदायिक कार्यको फाइदा के हुन्छ?",
    "नेपालका प्रदेशहरूको राजधानी कहाँ छ?",
    "सार्वजनिक स्थलहरूको सरसफाइ किन गर्नुपर्छ?",
]

# Streamlit 1.45+ supports st.pills
selected_pill = st.pills(
    "एउटा प्रश्न छान्नुहोस् वा तल टाइप गर्नुहोस्:",
    sample_questions,
    selection_mode="single",
)

# ─── 6. CHAT INPUT & RAG LOGIC ───
typed_prompt = st.chat_input("आफ्नो प्रश्न यहाँ लेख्नुहोस्... (Ask your question)")

# Use either the typed prompt or the clicked pill
prompt = typed_prompt if typed_prompt else selected_pill

if prompt:
    # Prevent re-running the same pill if user clicks it again immediately
    if selected_pill and (not st.session_state.messages or st.session_state.messages[-1]["content"] != prompt):
        pass
        
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("खोज्दै... (Searching...)"):
            
            # Embed Query (E5 requires "query: " prefix)
            q_vec = embedder.encode([f"query: {prompt}"], normalize_embeddings=True)[0].tolist()
            
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

                if distance > threshold:
                    answer = f"माफ गर्नुहोस्, यो प्रश्नको उत्तर सामग्रीमा छैन। (Distance: {distance:.3f})"
                else:
                    answer = metadata.get("answer") or metadata.get("response") or "Answer format error."
                    source_info = {
                        "distance": distance,
                        "unit": metadata.get("unit", "N/A"),
                        "lesson": metadata.get("lesson", "N/A"),
                        "document": document_text
                    }

            st.markdown(answer)
            
            if source_info:
                with st.expander("🔍 View Source & Score"):
                    st.write(f"**Distance:** {source_info['distance']:.3f}")
                    st.write(f"**Unit:** {source_info['unit']}")
                    st.write(f"**Lesson:** {source_info['lesson']}")
                    st.write(f"**Raw Document:** {source_info['document']}")

            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer,
                "source": source_info
            })
