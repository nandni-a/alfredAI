# gcp_maps_utils.py

import httpx

BASE_URL = "https://live-map-service-343916782787.us-central1.run.app/api"

async def get_nearby_places(lat: float, lng: float, place_type: str):
    url = f"{BASE_URL}/nearby_places?lat={lat}&lng={lng}&type={place_type}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

async def get_route(origin: str, destination: str):
    url = f"{BASE_URL}/route?origin={origin}&destination={destination}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

def format_places(places):
    if not places:
        return "<p>No places found.</p>"

    html_output = "<div style='font-family:sans-serif;'>"
    for place in places:
        name = place.get("name", "Unknown")
        rating = place.get("rating", "N/A")
        user_ratings_total = place.get("user_ratings_total", 0)
        address = place.get("address", "No address")
        open_now = place.get("open_now", None)
        photo_url = place.get("photo", "https://via.placeholder.com/400x200.png?text=No+Image")
        maps_link = place.get("maps_link", "https://www.google.com/maps")

        stars = "⭐" * int(float(rating)) if rating != "N/A" else "N/A"

        html_output += f"""
        <div style='margin-bottom:20px;padding:10px;border:1px solid #ccc;border-radius:12px;
        box-shadow:2px 2px 6px rgba(0,0,0,0.1);'>
            <h3><a href="{maps_link}" target="_blank" style="text-decoration:none;">{name}</a></h3>
            <img src="{photo_url}" style="max-width:100%;height:auto;border-radius:10px;">
            <p><b>Address:</b> {address}</p>
            <p><b>Rating:</b> {stars} ({rating} based on {user_ratings_total} reviews)</p>
            <p><b>Status:</b> {"🟢 Open Now" if open_now else "🔴 Closed" if open_now == False else "Unknown"}</p>
        </div>
        """
    html_output += "</div>"
    return html_output

def format_places_html(places, lat=None, lng=None, place_type=None):
    content = format_places(places)
    if lat and lng and place_type:
        map_link = f"https://live-map-service-343916782787.us-central1.run.app/?origin={lat},{lng}&type={place_type}"
        content += f'<br><a href="{map_link}" target="_blank">🗺️ Open in Map</a>'
    return content

def format_route(data):
    return f"Distance: {data['distance']['text']}\nDuration: {data['duration']['text']}"
