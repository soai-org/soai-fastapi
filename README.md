# soai-fastapi

## 📝 프로젝트 개요

본 프로젝트는 의료 영상 분석을 위한 다양한 AI 모델을 API 형태로 제공하는 FastAPI 기반의 서버입니다. 딥러닝 모델을 활용하여 충수염 진단, 의료 영상 분할, 영상 캡셔닝, 그리고 의료 관련 질문에 답변하는 챗봇 기능을 제공합니다.

## 🚀 주요 기능

- **충수염 진단:** DICOM 이미지로부터 충수염 발병 확률을 예측합니다.
- **이미지 분할:** 의료 영상에서 특정 영역을 분할(Segmentation)하여 시각화합니다.
- **이미지 메타데이터 분석 및 캡셔닝:** DICOM 이미지와 메타데이터를 분석하여 진단 결과를 텍스트로 생성합니다.
- **의료 챗봇:** 미세조정된 Gemma 모델을 사용하여 의료 관련 질문에 답변합니다.
- **실시간 통신:** WebSocket을 지원하여 실시간으로 모델의 추론 결과를 스트리밍합니다.

## 🛠️ 기술 스택

- **백엔드:** FastAPI, Uvicorn
- **AI/ML:** PyTorch, Transformers, PEFT, Scikit-learn
- **기타:** Pydicom, OpenCV, Pillow, python-dotenv

## ⚙️ 설치 및 실행

### 1. 프로젝트 클론

```bash
git clone https://github.com/soai-org/soai-fastapi.git
cd soai-fastapi
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\activate  # Windows
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env.sample` 파일을 복사하여 `.env` 파일을 생성하고, 아래와 같이 필요한 환경 변수를 설정합니다.

```.env
HF_TOKEN="your_huggingface_token"
ORTHANC_URL="http://your_orthanc_server_url"
ORTHANC_USERNAME="your_orthanc_username"
ORTHANC_PASSWORD="your_orthanc_password"
```

### 5. 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

서버가 정상적으로 실행되면 `http://localhost:8000/docs`에서 API 문서를 확인할 수 있습니다.

## 🔗 API 엔드포인트

### 💬 챗봇 (Chat-Bot)

- `POST /chat-bot/ask-llm`: Fine-tuning된 Gemma-3-1b 모델에게 질문하고 답변을 받습니다.
- `WS /chat-bot/ws-llm`: WebSocket을 통해 실시간으로 질의응답을 수행합니다.

### 🖼️ 이미지 분할 (Image Segmentation)

- `POST /image/segmentation`: DICOM 이미지의 Instance UUID를 받아 분할된 이미지를 반환합니다.
- `POST /image/segmentation_array`: 분할된 이미지의 배열 값을 반환합니다.
- `WS /image/ws-segmentation`: WebSocket을 통해 실시간으로 이미지 분할을 수행하고 결과를 스트리밍합니다.

### 🩺 이미지 메타데이터 진단 (Image-Meta Diagnosis)

- `POST /image-meta/diagnosis`: 환자의 메타데이터와 이미지를 분석하여 진단 결과를 영문 텍스트로 생성합니다.
- `WS /image-meta/ws-diagnosis`: WebSocket을 통해 실시간으로 진단 결과를 스트리밍합니다.

### 🩹 충수염 진단 (Appendicitis Diagnosis)

- `POST /appendicitis/diagnosis`: DICOM 이미지들의 Instance UUID 리스트를 받아 충수염 진단 결과를 반환합니다.

## 🤖 모델

본 프로젝트에서 사용하는 주요 모델은 다음과 같습니다.

- **`appendicitis_model.pth`**: 충수염 진단을 위한 Multi-view 모델
- **`image_meta.pth`**: 이미지 캡셔닝을 위한 Vision-NLP 모델
- **`Segmentation.pth`**: 이미지 분할을 위한 U-Net 기반 모델
- **`resnet18-5c106cde.pth`**: 이미지 처리에 사용되는 ResNet18 모델
- **Fine-tuned Gemma-3-1b**: 의료 질문에 답변하기 위해 미세조정된 언어 모델

## 📁 디렉터리 구조

```
.
├── .env.sample                  # 환경 변수 설정 예시 파일
├── .gitignore                   # Git 버전 관리에서 제외할 파일 목록
├── main.py                      # FastAPI 애플리케이션의 메인 실행 파일
├── README.md                    # 프로젝트 설명서
├── requirements.txt             # 프로젝트에 필요한 Python 패키지 목록
├── app
│   ├── core
│   │   └── config.py            # 애플리케이션 환경 변수 및 설정 관리
│   ├── models
│   │   ├── appendicitis_model.pth # 충수염 진단 모델 가중치
│   │   ├── image_meta.pth       # 이미지 캡셔닝 모델 가중치
│   │   ├── resnet18-5c106cde.pth  # ResNet18 모델 가중치
│   │   └── Segmentation.pth     # 이미지 분할 모델 가중치
│   ├── routers
│   │   ├── appendicitis_model.py # 충수염 진단 API 라우터
│   │   ├── finetuned_gemma.py    # Fine-tuned Gemma 챗봇 API 라우터
│   │   ├── image_meta.py         # 이미지 메타데이터 분석 API 라우터
│   │   ├── segmentation_model.py # 이미지 분할 API 라우터
│   │   ├── ws_finetuned_gemma.py # Gemma 챗봇 WebSocket API 라우터
│   │   └── ws_nlp_streamer.py    # NLP 스트리밍 WebSocket API 라우터
│   ├── schema
│   │   ├── appendicitis_schema.py # 충수염 진단 API의 요청/응답 스키마
│   │   ├── chat_schema.py         # 챗봇 API의 요청/응답 스키마
│   │   ├── image_meta_schema.py   # 이미지 메타데이터 API의 요청/응답 스키마
│   │   └── segmentation_request_schema.py # 이미지 분할 API의 요청/응답 스키마
│   └── services
│       ├── appendicitis_network.py # 충수염 진단 모델의 네트워크 구조 정의
│       ├── finetuned_llm.py        # Fine-tuned LLM 관련 비즈니스 로직
│       ├── Multiview.py            # Multi-view 이미지 처리 로직
│       ├── parsing_patientdata.py  # 환자 데이터 파싱 로직
│       ├── segmentation.py         # 이미지 분할 관련 비즈니스 로직
│       ├── vis_nlp.py              # Vision-NLP 모델 관련 비즈니스 로직
│       ├── ws_llm_streamer.py      # LLM 스트리밍 WebSocket 서비스
│       └── ws_nlp_streamer.py      # NLP 스트리밍 WebSocket 서비스
└── rest
    └── Dicom
        └── Dicom.ipynb              # DICOM 파일 테스트 및 실험용 Jupyter Notebook
```
