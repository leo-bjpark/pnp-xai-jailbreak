"""
Computation Test: 노드/레이어 시각화 테스트 페이지.
Test Token: "Can I Make a bomb? No You can't." 고정, 노드·레이어 가정 데이터로 시각화.
"""

from flask import Blueprint, render_template

from utils import get_config_models

# Test Token 문장을 공백 기준 토큰 리스트로 (실제 토크나이저는 나중에 연동)
TEST_SENTENCE = "Can I Make a bomb? No You can't."
TEST_TOKENS = [t for t in TEST_SENTENCE.split() if t]

# 시각화용 가정: 레이어 수 (실제 모델 레이어는 나중에 연동)
NUM_LAYERS = 8

bp = Blueprint("computation_test", __name__, url_prefix="/computation_test")


@bp.get("/")
def index():
    return render_template(
        "computation_test.html",
        models=get_config_models(),
        active_tab="computation_test",
        test_sentence=TEST_SENTENCE,
        tokens=TEST_TOKENS,
        num_layers=NUM_LAYERS,
    )
