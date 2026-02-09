# Pnp-XAI-LLM
 

## XAI-Levels 

- Level 0 : 
- Level 1 : 
- Level 2 : 


## Code Design / Structure 

### High-level layout

- **`app.py` (entrypoint)**: 최소 역할만 담당하는 엔트리 파일.
  - `from python.web import create_app` 호출
  - `app = create_app()` 생성
  - `if __name__ == "__main__": app.run(...)` 만 유지

- **`python/web/` (Flask 웹 레이어)**
  - **역할**: HTTP 요청/응답, JSON 파싱, 세션/권한 체크, 템플릿 렌더링, SSE 스트림 등 “웹/컨트롤러” 로직.
  - **구성 (제안)**:
    - `python/web/__init__.py`
      - `create_app()` 앱 팩토리
      - Blueprint 등록 (`main_bp`, `tasks_bp`, `session_bp`, `run_bp`, `memory_bp`, `dataset_bp`, `residual_bp` 등)
    - `python/web/views_main.py`
      - `/`, `/panel`, `/task/<task_id>`, `/data`, `/data/<pipeline_id>` 등 템플릿 렌더링 라우트
    - `python/web/api_tasks.py`
      - `/api/tasks*` (생성/조회/수정/삭제)
    - `python/web/api_session.py`
      - `/api/session*`, `/api/models`, `/api/load_model`, `/api/model_status`, `/api/cuda_env*`
    - `python/web/api_run.py`
      - `/api/run`, `/api/run/residual-concept-stream`
    - `python/web/api_memory.py`
      - `/api/memory/*`, `/api/empty_cache`
    - `python/web/api_dataset.py`
      - `/api/dataset-pipelines*`, `/api/data-vars*`
    - `python/web/api_residual.py`
      - `/api/residual-vars*`

- **`python/xai_handlers/` (기존 `python/routes/` → 이름 변경)**  
  - **역할**: XAI 레벨별 비즈니스 로직. Flask/HTTP를 모르고, 순수 Python 서비스 함수만 제공.
  - **파일 매핑 (이름 변경)**:
    - `python/routes/xai_0.py` → `python/xai_handlers/level_0.py`
      - `run_conversation(...)`, `run_completion(...)`
    - `python/routes/xai_1.py` → `python/xai_handlers/level_1.py`
      - `run_attribution(...)`
    - `python/routes/xai_2.py` → `python/xai_handlers/level_2.py`
      - `run_residual_concept(...)`, `run_placeholder(...)`
    - `python/routes/__init__.py` → `python/xai_handlers/__init__.py`
      - 각 레벨 핸들러 re-export 용 (`from .level_0 import run_conversation, ...`)
  - **사용 방식**:
    - 웹 레이어(`python/web/api_run.py`)에서 `from python.xai_handlers import run_conversation` 처럼 import 한 뒤,
    - HTTP 요청 파라미터를 검증/전처리 → 핸들러 함수 호출 → 결과 JSON 을 그대로 응답.

- **`python/services/` (공통 서비스 / 유틸 계층, 이름만 정의)**  
  - **역할**: 모델 구조 분석, 데이터셋 로딩/샘플링/가공, 메모리/캐시 관리 등 웹/GUI 에 독립적인 “도메인 서비스”.
  - **예시 모듈 (제안)**:
    - `python/services/model_introspection.py`
      - `app.py`의 `_detect_layer_structure`, `_empty_layer_structure` 및 `/api/model_layer_names` 로직 분리
    - `python/services/dataset_service.py`
      - `_random_select_dataset`, `_dataset_to_info`, `_safe_value`, `_load_pipeline_dataset`, `_get_process_function`
    - 웹 레이어와 XAI 핸들러가 공통으로 사용하는 기능은 이 계층에 둔다.

### Memory / Task 개념 (요약)

|Icon | Name           | Type           | Management             |
|-----|----------------|----------------|------------------------|
| 🧵  | Task Session   | Python Cache   | Alive & Terminated     |
| 📄  | Task Result    | Json Format    | Save & Load & Delete   |
| 🧊  | Variables      | Python Address | Save & Load & Delete   |

- **Task Panel**
  - Cache Memory (Session Memory)
  - Stored Memory (Json Format)

- **Working Memory**
  - Python 객체를 Variable 로 저장/불러오기
  - XAI Task Panels, Data Processing Panels 에서 공통으로 활용