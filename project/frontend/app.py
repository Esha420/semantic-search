import streamlit as st
import requests

st.title("🔍 eSewa AI Search")

query = st.text_input("Enter your search query:")

if st.button("Search") and query:
    with st.spinner("Searching..."):
        try:
            response = requests.get("http://localhost:8001/search", params={"q": query})
            results = response.json().get("results", [])
            if results:
                st.success(f"Top {len(results)} results:")
                for res in results:
                    st.json(res)
            else:
                st.warning("No results found.")
        except Exception as e:
            st.error(f"Error: {e}")
