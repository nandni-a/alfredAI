# streamlit_ui.py

import os
import time
import asyncio
import streamlit as st
from langchain_core.messages import HumanMessage
from base_reasoning import get_llm, query_collection, create_inmemory_collection
from gcp_maps_utils import get_nearby_places, get_route, format_places, format_route
from st_audiorec import st_audiorec
import whisper

st.set_page_config(page_title="Alfred Assistant", page_icon="🤖")
st.title("🤖 Alfred Assistant")
st.caption("Multi-LLM Assistant with Voice, Memory, Location & File Q&A")

# Load LLM and Whisper
llm = get_llm("openai")
whisper_model = whisper.load_model("base")
collection = create_inmemory_collection()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for i, msg in enumerate(st.session_state.messages):
    st.chat_message(msg["role"]).write(msg["content"])

# Voice input
st.markdown("#### 🎙️ Voice Input (optional)")
wav_audio = st_audiorec()
user_input = ""
if wav_audio is not None:
    with open("audio.wav", "wb") as f:
        f.write(wav_audio)
    transcript = whisper_model.transcribe("audio.wav")
    user_input = transcript["text"]
    st.success(f"You said: {user_input}")

# Manual input fallback
text_input = st.chat_input("Type your message")
if text_input:
    user_input = text_input

# Chat handling
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke([HumanMessage(content=user_input)])
            reply = getattr(response, "content", str(response))
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

st.divider()
st.markdown("### 📄 Ask from Uploaded File")
file = st.file_uploader("Upload a .txt file", type=["txt"])
question = st.text_input("Ask something about the uploaded file")
if st.button("Ask about file") and file and question:
    content = file.read().decode('utf-8')
    collection.add(documents=[content], metadatas=[{"source": file.name}], ids=[file.name])
    context = query_collection(collection, question)
    if context:
        file_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        file_response = llm.invoke([HumanMessage(content=file_prompt)])
        st.markdown("**Answer:**")
        st.info(getattr(file_response, "content", str(file_response)))
    else:
        st.warning("No relevant content found in the uploaded file.")

st.divider()
st.markdown("### 📍 Location Tools")
tool = st.selectbox("Choose Location Tool", ["Nearby Places", "Route Info"])

if tool == "Nearby Places":
    lat = st.text_input("Latitude")
    lng = st.text_input("Longitude")
    place_type = st.text_input("Place Type (e.g. cafe, library)")
    if st.button("Find Nearby"):
        if lat and lng and place_type:
            try:
                places = asyncio.run(get_nearby_places(float(lat), float(lng), place_type))
                st.success("Nearby Places Found:")
                st.markdown(format_places(places))
            except Exception as e:
                st.error(f"Error: {e}")

elif tool == "Route Info":
    origin = st.text_input("Origin (lat,lng)")
    dest = st.text_input("Destination (lat,lng)")
    if st.button("Get Route"):
        if origin and dest:
            try:
                data = asyncio.run(get_route(origin, dest))
                st.success("Route Info:")
                st.markdown(format_route(data))
                map_url = f"https://live-map-service-343916782787.us-central1.run.app/?origin={origin}&destination={dest}"
                st.markdown(f"[🔗 View Route Map]({map_url})")
            except Exception as e:
                st.error(f"Error: {e}")