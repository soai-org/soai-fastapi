# 📂 Project Structure
app/

├── models/

│ ├── image_meta.pth

│ └── Segmentation.pth

│

├── routers/

│ ├── finetuned_gemma.py

│ ├── image_meta.py

│ └── segmentation_model.py

│

├── schema/

│ ├── chat_schema.py

│ ├── image_meta_schema.py

│ └── segmentation_request_schema.py

│

└── services/

  ├── gemma-3-1b-it-finetuned-final/
  
  ├── init.py
  
  ├── finetuned_llm.py
  
  ├── parsing_patientdata.py
  
  ├── segmentation.py
  
  └── vis_nlp.py

  ## 📌 설명
- **models/**  
  - `image_meta.pth`: 이미지 메타데이터 관련 모델 가중치  
  - `Segmentation.pth`: 세그멘테이션 모델 가중치  

- **routers/**  
  - API 엔드포인트 정의 (이미지 메타, 세그멘테이션, 파인튜닝 모델 관련)  

- **schema/**  
  - 요청/응답을 위한 데이터 스키마 정의 (pydantic 기반)  

- **services/**  
  - 비즈니스 로직, 모델 로딩 및 전처리/후처리  
  - `segmentation.py`: 세그멘테이션 서비스  
  - `finetuned_llm.py`: LLM 파인튜닝 처리  
  - `vis_nlp.py`: NLP 시각화 관련 로직  
