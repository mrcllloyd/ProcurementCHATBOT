import streamlit as st
from chatbot_engine import semantic_search, ask_chatbot

st.set_page_config("ProcureBot PH", layout="wide")
st.title("📜 ProcureBot PH")
st.caption("Ask questions about the Government Procurement Law (RA 12009) and its IRR.")

query = st.text_input("🔍 Ask your procurement law question:")

if query:
    with st.spinner("🔎 Searching RA 12009 and its IRR..."):
        results = semantic_search(query)
        context = [text for _, text in results]
        answer = ask_chatbot(query, context)

    st.subheader("🤖 ProcureBot's Answer")
    st.write(answer)

    with st.expander("📚 Legal References Used"):
        for i, (src, txt) in enumerate(results, 1):
            st.markdown(f"**{i}. Source: `{src}`**\n\n> {txt}\n\n---")
