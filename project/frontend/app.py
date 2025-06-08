import streamlit as st
import requests

st.title("🔍 Semantic Search UI")

query = st.text_input("Enter your search query:")

if st.button("Search") and query:
    response = requests.get("http://localhost:8000/search", params={"query": query})
    results = response.json()

    st.subheader("Results:")
    for res in results:
        st.markdown(f"**Score:** {res['score']:.4f}")
        st.markdown(f"```{res['text']}```")
