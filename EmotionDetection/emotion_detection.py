"""
Emotion detection module using Watson NLP (embedded service on Skills Network).
"""

import json
from typing import Any, Dict, Optional

import requests

URL = (
    "https://sn-watson-emotion.labs.skills.network/v1/"
    "watson.runtime.nlp.v1/NlpService/EmotionPredict"
)
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}


def emotion_detector(text_to_analyze: str) -> Dict[str, Optional[float]]:
    """
    Detect emotions from input text using Watson NLP EmotionPredict endpoint.

    Args:
        text_to_analyze: Input text from the user.

    Returns:
        Dictionary containing emotion scores and dominant emotion.
        For invalid/blank input (status_code=400), returns keys with None values.
    """
    payload: Dict[str, Any] = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(URL, json=payload, headers=HEADERS, timeout=30)

    # Error handling for blank input
    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None,
        }

    response_text = response.text
    data = json.loads(response_text)

    # Expected structure:
    # data["emotionPredictions"][0]["emotion"]
    emotions = data["emotionPredictions"][0]["emotion"]

    anger = emotions["anger"]
    disgust = emotions["disgust"]
    fear = emotions["fear"]
    joy = emotions["joy"]
    sadness = emotions["sadness"]

    scores = {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
    }

    dominant_emotion = max(scores, key=scores.get)

    return {
        "anger": anger,
        "disgust": disgust,
        "fear": fear,
        "joy": joy,
        "sadness": sadness,
        "dominant_emotion": dominant_emotion,
    }
