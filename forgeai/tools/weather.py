"""Current conditions + short forecast from open-meteo.com (keyless public API)."""

from __future__ import annotations

import requests

ENDPOINT = "https://api.open-meteo.com/v1/forecast"

_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 61: "slight rain", 63: "rain", 65: "heavy rain",
    71: "slight snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "moderate rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}


def run(latitude=None, longitude=None, place: str = "", **_ignored) -> str:
    if latitude is None or longitude is None:
        return "ERROR: latitude and longitude are required"
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError):
        return "ERROR: latitude and longitude must be numbers"
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return "ERROR: coordinates out of range"

    try:
        resp = requests.get(
            ENDPOINT,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,weather_code",
                "forecast_days": 3,
                "timezone": "UTC",
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"ERROR: weather lookup failed: {exc}"

    label = place or f"{lat:.2f},{lon:.2f}"
    cur = data.get("current", {})
    lines = [
        f"Weather for {label} (open-meteo, UTC):",
        f"- now: {cur.get('temperature_2m')}degC, wind {cur.get('wind_speed_10m')} km/h, "
        f"{_CODES.get(cur.get('weather_code'), 'unknown conditions')}",
    ]
    daily = data.get("daily", {})
    for i, day in enumerate(daily.get("time", [])[:3]):
        lines.append(
            f"- {day}: {daily['temperature_2m_min'][i]} to "
            f"{daily['temperature_2m_max'][i]} degC, "
            f"{_CODES.get(daily['weather_code'][i], 'unknown')}"
        )
    return "\n".join(lines)
