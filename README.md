# 📂 Project Structure
app/

├── models/

│ ├── image_meta.pth # 이미지 + 메타데이터 진단(Captioning) 모델의 가중치

│ └── Segmentation.pth # 이미지 세그멘테이션 모델의 가중치

│

├── routers/

│ ├── finetuned_gemma.py # Gemma3 1b 파인 튜닝 모델 관련 API 라우터

│ ├── image_meta.py # 이미지 + 메타데이터 캡셔닝 모델 관련 API 라우터

│ └── segmentation_model.py # 세그멘테이션 모델 API 라우터

│

├── schema/

│ ├── chat_schema.py # 채팅 관련 요청/응답 스키마

│ ├── image_meta_schema.py # 이미지 메타데이터 요청/응답 스키마

│ └── segmentation_request_schema.py # 이미지 메타데이터 요청/응답 스키마

│

└── services/

│  ├── gemma-3-1b-it-finetuned-final/ # Gemma 모델 최종 fine-tuned 폴더
  
│  ├── finetuned_llm.py # Fine-tuned LLM 비즈니스 로직 처리
  
│  ├── parsing_patientdata.py # 환자 데이터 파싱 서비스
  
│  ├── segmentation.py # 세그멘테이션 비즈니스 로직 처리
  
│  └── vis_nlp.py # 이미지 + 메타데이터 모델 비즈니스 로직 처리

└── main.py # 최종 실행 파일

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
  - `vis_nlp.py`: 이미지 + 메타데이터 모델 비즈니스 로직 처리
