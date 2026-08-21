import streamlit as st
import requests

API_INTERNAL_URL = "http://api:8000"
API_EXTERNAL_URL = "http://localhost:8000"

st.set_page_config(layout="wide", page_title="Multimodal Search")
st.title("Multimodal Search Engine")

with st.sidebar:
    st.header("1. Image Indexing")
    uploaded_index_file = st.file_uploader("Upload an image to index", type=["jpg", "jpeg", "png"], key="index_upload")

    if st.button("Index Image", use_container_width=True):
        if uploaded_index_file:
            with st.spinner("Processing..."):
                files = {"file": (uploaded_index_file.name, uploaded_index_file.getvalue(), uploaded_index_file.type)}
                response = requests.post(f"{API_INTERNAL_URL}/ingest/image", files=files)
                if response.status_code == 200:
                    st.success("Indexed successfully!")
                else:
                    st.error(f"Error: {response.text}")
        else:
            st.warning("Upload a file first.")

    st.divider()

    st.header("2. System Status")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Health Check", use_container_width=True):
            res = requests.get(f"{API_INTERNAL_URL}/health")
            st.json(res.json()) if res.status_code == 200 else st.error("Offline")
    with col2:
        if st.button("DB Stats", use_container_width=True):
            res = requests.get(f"{API_INTERNAL_URL}/stats")
            st.json(res.json()) if res.status_code == 200 else st.error("Error fetching stats")


def render_results(results):
    if not results:
        st.info("No results found.")
        return

    cols = st.columns(4)
    for idx, res in enumerate(results):
        payload_data = res.get("payload", {})
        image_path = payload_data.get("image_url") or res.get("image_url")
        score = res.get("score", 0.0)
        point_id = res.get("id", "Unknown ID")

        with cols[idx % 4]:
            if image_path:
                full_image_url = f"{API_EXTERNAL_URL}{image_path}"
                st.image(full_image_url, caption=f"Score: {score:.3f} | ID: {point_id[:8]}...", width="stretch")
            else:
                st.warning(f"Imagine fără URL (ID: {point_id})")

tab_text, tab_image, tab_manage = st.tabs(["Search by Text", "Search by Image", "Manage Database"])

with tab_text:
    query = st.text_input("Enter search query", key="text_query")
    limit_text = st.slider("Number of results", 1, 12, 4, key="limit_text")

    if st.button("Search Text"):
        if query:
            with st.spinner("Searching..."):
                payload = {"query": query, "limit": limit_text}
                response = requests.post(f"{API_INTERNAL_URL}/search/text", data=payload)
                if response.status_code == 200:
                    render_results(response.json())
                else:
                    st.error(f"Error: {response.text}")

with tab_image:
    search_image_file = st.file_uploader("Upload an image to search visually similar ones", type=["jpg", "jpeg", "png"],
                                         key="search_upload")
    limit_img = st.slider("Number of results", 1, 12, 4, key="limit_img")

    if st.button("Search Image"):
        if search_image_file:
            with st.spinner("Searching..."):
                # FastApi așteaptă 'file' (UploadFile) și 'limit' (Form)
                files = {"file": (search_image_file.name, search_image_file.getvalue(), search_image_file.type)}
                data = {"limit": limit_img}
                response = requests.post(f"{API_INTERNAL_URL}/search/image", files=files, data=data)

                if response.status_code == 200:
                    render_results(response.json())
                else:
                    st.error(f"Error: {response.text}")
        else:
            st.warning("Upload an image first.")

with tab_manage:
    st.subheader("Delete Points")
    ids_to_delete = st.text_area("Enter UUIDs to delete (comma separated)",
                                 placeholder="e.g. 123e4567-e89b-12d3-a456-426614174000")

    if st.button("Delete specific vectors"):
        if ids_to_delete:
            # Curățăm string-ul și facem lista de ID-uri
            id_list = [uuid.strip() for uuid in ids_to_delete.split(",") if uuid.strip()]
            with st.spinner("Deleting..."):
                response = requests.post(f"{API_INTERNAL_URL}/delete/points", json=id_list)
                if response.status_code == 200:
                    st.success(f"Successfully deleted {len(id_list)} vectors.")
                else:
                    st.error(f"Error: {response.text}")