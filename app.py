"""Streamlit interface for the Enterprise Document QA API."""

import requests
import streamlit as st

from app.config import DEFAULT_MIN_SCORE, DEFAULT_TOP_K


API_BASE_URL = "http://127.0.0.1:8000"
HEALTH_TIMEOUT_SECONDS = 10
REQUEST_TIMEOUT_SECONDS = 60
INDEX_TIMEOUT_SECONDS = 900


def _error_detail(response: requests.Response) -> str:
    """Return an understandable backend error without exposing internals."""

    try:
        detail = response.json().get("detail")
    except (requests.JSONDecodeError, AttributeError, ValueError):
        detail = None

    if isinstance(detail, str) and detail.strip():
        return detail
    if isinstance(detail, list):
        messages = [
            str(item.get("msg"))
            for item in detail
            if isinstance(item, dict) and item.get("msg")
        ]
        if messages:
            return "; ".join(messages)
    return f"Backend request failed with status {response.status_code}."


def _render_sources(sources: list[dict[str, object]]) -> None:
    """Render only source citations returned by the backend."""

    st.markdown("**Sources**")
    if not sources:
        st.caption("No supporting sources found.")
        return

    for source in sources:
        st.write(
            f"[{source['citation_number']}] {source['document_name']} "
            f"— Page {source['page_number']}"
        )


def _render_retrieval(retrieval: list[dict[str, object]]) -> None:
    """Render compact retrieval metadata without displaying chunk text."""

    with st.expander("Retrieval details"):
        if not retrieval:
            st.caption("No retrieval results passed the similarity threshold.")
            return

        rows = [
            {
                "Rank": rank,
                "Similarity score": round(float(result["score"]), 4),
                "Document": result["document_name"],
                "Page": result["page_number"],
                "Chunk ID": result["chunk_id"],
            }
            for rank, result in enumerate(retrieval, start=1)
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_assistant_message(message: dict[str, object]) -> None:
    """Render one stored assistant answer and its supporting metadata."""

    st.markdown(str(message["content"]))
    _render_sources(list(message.get("sources", [])))
    _render_retrieval(list(message.get("retrieval", [])))


st.set_page_config(
    page_title="Enterprise Document QA",
    page_icon="📄",
    layout="wide",
)

st.title("Enterprise Document QA")
st.write(
    "Ask grounded questions across your indexed documents using "
    "retrieval-augmented generation."
)
st.caption(
    "Semantic retrieval • Source citations • Unsupported-query handling"
)

if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("Documents")

    backend_connected = False
    indexed_vectors = 0
    try:
        health_response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=HEALTH_TIMEOUT_SECONDS,
        )
        if health_response.ok:
            health_data = health_response.json()
            backend_connected = health_data.get("status") == "ok"
            indexed_vectors = int(health_data.get("indexed_vectors", 0))
    except (requests.RequestException, ValueError, TypeError):
        backend_connected = False

    if backend_connected:
        st.success("Backend: Connected")
    else:
        st.error("Backend: Unavailable")
    vector_display = st.empty()
    vector_display.metric("Indexed vectors", indexed_vectors)

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button(
        "Upload PDFs",
        use_container_width=True,
        disabled=not backend_connected or not uploaded_files,
    ):
        multipart_files = [
            (
                "files",
                (uploaded_file.name, uploaded_file.getvalue(), "application/pdf"),
            )
            for uploaded_file in uploaded_files
        ]
        try:
            with st.spinner("Uploading PDFs..."):
                upload_response = requests.post(
                    f"{API_BASE_URL}/documents/upload",
                    files=multipart_files,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            if upload_response.ok:
                saved_files = upload_response.json().get("filenames", [])
                st.success(f"Uploaded: {', '.join(saved_files)}")
            else:
                st.error(_error_detail(upload_response))
        except requests.RequestException:
            st.error("The PDFs could not be uploaded because the backend is unavailable.")

    if st.button(
        "Index / Rebuild Documents",
        use_container_width=True,
        disabled=not backend_connected,
    ):
        try:
            with st.spinner("Indexing documents. This may take a few minutes..."):
                index_response = requests.post(
                    f"{API_BASE_URL}/documents/index",
                    timeout=INDEX_TIMEOUT_SECONDS,
                )
            if index_response.ok:
                index_summary = index_response.json()
                st.success("Document index rebuilt successfully.")
                st.write(f"PDFs indexed: {index_summary['pdf_count']}")
                st.write(f"Pages extracted: {index_summary['extracted_pages']}")
                st.write(f"Chunks created: {index_summary['chunks']}")
                st.write(f"Vectors stored: {index_summary['vectors']}")
                vector_display.metric("Indexed vectors", index_summary["vectors"])
            else:
                st.error(_error_detail(index_response))
        except requests.RequestException:
            st.error("Indexing could not start because the backend is unavailable.")

    document_names: list[str] = []
    if backend_connected:
        try:
            documents_response = requests.get(
                f"{API_BASE_URL}/documents",
                timeout=HEALTH_TIMEOUT_SECONDS,
            )
            if documents_response.ok:
                document_names = list(
                    documents_response.json().get("documents", [])
                )
            else:
                st.warning(_error_detail(documents_response))
        except (requests.RequestException, ValueError, TypeError):
            st.warning("The document list is currently unavailable.")

    st.subheader("Available PDFs")
    if document_names:
        for document_name in document_names:
            st.caption(f"• {document_name}")
    else:
        st.caption("No PDF documents found.")

    selected_document = st.selectbox(
        "Document filter",
        ["All documents", *document_names],
    )
    document_filter = (
        None if selected_document == "All documents" else selected_document
    )

    st.subheader("Retrieval settings")
    st.write(f"Top-K: {DEFAULT_TOP_K}")
    st.write(f"Minimum similarity score: {DEFAULT_MIN_SCORE:.2f}")
    st.caption("Defaults selected from retrieval evaluation experiments.")

    with st.expander("Advanced settings"):
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=8,
            value=DEFAULT_TOP_K,
            step=1,
        )
        min_score = st.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_MIN_SCORE,
            step=0.05,
            format="%.2f",
        )

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


for stored_message in st.session_state.messages:
    with st.chat_message(str(stored_message["role"])):
        if stored_message["role"] == "assistant":
            _render_assistant_message(stored_message)
        else:
            st.markdown(str(stored_message["content"]))


question = st.chat_input(
    "Ask a question about your documents",
    disabled=not backend_connected,
)

if question:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(question)

    query_payload = {
        "question": question,
        "top_k": top_k,
        "min_score": min_score,
        "category": None,
        "document_name": document_filter,
    }

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching indexed documents..."):
                query_response = requests.post(
                    f"{API_BASE_URL}/query",
                    json=query_payload,
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

            if query_response.ok:
                query_data = query_response.json()
                assistant_message = {
                    "role": "assistant",
                    "content": query_data["answer"],
                    "sources": query_data.get("sources", []),
                    "retrieval": query_data.get("retrieval", []),
                }
                _render_assistant_message(assistant_message)
                st.session_state.messages.append(assistant_message)
            else:
                st.error(_error_detail(query_response))
    except (requests.RequestException, ValueError, TypeError, KeyError):
        st.error("The question could not be completed. Check the backend and try again.")
