"""Brain concept map API for 2.1.0 Brain Concept Visualization."""

from flask import Blueprint, jsonify

from python.brain_config import load_brain_nodes

brain_bp = Blueprint("brain", __name__)


@brain_bp.get("/api/brain-config")
def api_brain_config():
    """Return brain concept map nodes from brain.yaml."""
    nodes = load_brain_nodes()
    return jsonify({"nodes": nodes})
