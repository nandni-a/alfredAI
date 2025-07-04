# alfred_gradio_app.py

import os
import gradio as gr
import asyncio
import requests
import speech_recognition as sr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from base_reasoning import get_llm, query_collection, create_inmemory_collection
from gcp_maps_utils import get_nearby_places, get_route, format_places, format_route
from langchain_core.messages import HumanMessage

os.environ["CHROMA_TELEMETRY_ENABLED"] = "False"

# ---- Initialize ----
llm = get_llm("openai")
collection = create_inmemory_collection()
chat_memory = []
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")

# Image Captioning Model (BLIP)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# ---- Core Functions ----
def chat_with_memory(user_input, history):
    chat_memory.append({"role": "user", "content": user_input})
    messages = [HumanMessage(content=msg["content"]) for msg in chat_memory if msg["role"] == "user"]
    response = llm.invoke(messages)
    chat_memory.append({"role": "assistant", "content": response.content})
    return chat_memory

def respond_voice(audio_file, history):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
    return "", chat_with_memory(text, history)

def upload_and_ask(file, question):
    content = file.read().decode('utf-8')
    collection.add(documents=[content], metadatas=[{"source": file.name}], ids=[file.name])
    context = query_collection(collection, question)
    if not context:
        return "No relevant content found."
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

async def async_nearby(lat, lng, place_type):
    places = await get_nearby_places(float(lat), float(lng), place_type)
    return format_places(places)

def generate_map_url(lat, lng, place_type):
    return f"https://live-map-service-343916782787.us-central1.run.app/?origin={lat},{lng}&type={place_type}"

def fetch_route(origin, destination):
    try:
        route = asyncio.run(get_route(origin, destination))
        directions = format_route(route)
        map_url = f"https://live-map-service-343916782787.us-central1.run.app/?origin={origin}&destination={destination}"
        return directions, f"<a href='{map_url}' target='_blank'>🔗 View Route Map</a>"
    except Exception as e:
        return f"Error: {e}", ""

def serp_search(query):
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": SERP_API_KEY, "engine": "google"}
    response = requests.get(url, params=params)
    data = response.json()
    results = data.get("organic_results", [])
    html = ""
    for r in results[:5]:
        title = r.get("title", "No title")
        link = r.get("link", "#")
        snippet = r.get("snippet", "")
        displayed_link = r.get("displayed_link", link)
        html += f"""
        <div style='margin-bottom:1em;border-bottom:1px solid #ddd;padding-bottom:1em;'>
            <a href='{link}' target='_blank' style='font-size:1.1em;font-weight:bold;'>{title}</a><br>
            <span style='color:gray;font-size:0.9em;'>{displayed_link}</span><br>
            <p style='margin-top:4px;'>{snippet}</p>
        </div>
        """
    return html or "No results found."

def describe_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return f"🎼 Image describes: {caption}", chat_with_memory(caption, chat_memory)

# ---- Gradio UI ----
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Alfred Assistant — Smart Chat, Maps, Web & Image Q&A")
    chatbot = gr.Chatbot(label="Chat with Alfred", type="messages")
    with gr.Row():
        msg = gr.Textbox(label="Type your message")
        audio = gr.Audio(label="🎤 Use voice", format="wav")
        image = gr.Image(label="📸 Upload image", type="filepath")
        file = gr.File(label="📄 Upload .txt file")
    with gr.Row():
        send_btn = gr.Button("Send")
        clear_btn = gr.Button("Clear")
    html_output = gr.HTML(label="Results")

    def smart_router(user_msg, history):
        user_msg_lower = user_msg.lower()
        if any(x in user_msg_lower for x in ["cafes near", "restaurant near", "places near", "near me"]):
            lat, lng = 28.6139, 77.2090  # Default to Delhi if GPS not implemented
            html = asyncio.run(async_nearby(lat, lng, "cafe"))
            link = generate_map_url(lat, lng, "cafe")
            full_html = html + f"<br><br><a href='{link}' target='_blank'>🗺️ Open in Map</a>"
            return "", history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": full_html}]
        elif any(x in user_msg_lower for x in ["route to", "directions to", "navigate to"]):
            if "from" in user_msg_lower and "to" in user_msg_lower:
                parts = user_msg_lower.split("to")
                origin = parts[0].replace("from", "").strip()
                destination = parts[1].strip()
                directions, map_link = fetch_route(origin, destination)
                combined = f"<pre>{directions}</pre><br>{map_link}"
                return "", history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": combined}]
        else:
            html = serp_search(user_msg)
            return "", history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": html}]

    send_btn.click(smart_router, inputs=[msg, chatbot], outputs=[msg, chatbot])
    audio.change(respond_voice, inputs=[audio, chatbot], outputs=[msg, chatbot])
    image.change(describe_image, inputs=image, outputs=[msg, chatbot])
    clear_btn.click(lambda: [], outputs=[chatbot])

if __name__ == '__main__':
    demo.launch()
