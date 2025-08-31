import requests
from requests.auth import HTTPBasicAuth
import io
import numpy as np
from typing import Tuple, Dict, List, Any
import torch
import os
from dotenv import load_dotenv

ORTHANC_URL = os.getenv("ORTHANC_URL")

class PatientData:
    def __init__(self, ORTHANC_URL: str) -> None:
        self.ORTHANC_URL: str = ORTHANC_URL

        username = os.getenv("ORTHANC_USERNAME")
        password = os.getenv("ORTHANC_PASSWORD")

        basic_auth = HTTPBasicAuth(username, password)
        self.all_patients: List[str] = requests.get(f"{self.ORTHANC_URL}/patients", auth=basic_auth).json()

    def parsing_target_patient(self, PATIENT_NAME: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        찾고자하는 환자의 메타 데이터 및 Studies UUIDs 추출

        Args:
            PATIENT_NAME (str): 검색할 환자 이름

        Returns:
            Tuple[Dict[str, Any], List[str]]: 
                - 환자 메인 태그 정보 (PatientID, PatientName, PatientSex 등)
                - 해당 환자의 Study UUID 리스트
        """
        for uuid in self.all_patients:
            patients_info = requests.get(f"{self.ORTHANC_URL}/patients/{uuid}").json()
            if patients_info.get("MainDicomTags", {}).get("PatientName") == PATIENT_NAME:
                patient_tag_data: Dict[str, Any] = patients_info.get("MainDicomTags", {})
                studies: List[str] = patients_info.get("Studies", [])
                return patient_tag_data, studies
        return f"Patient '{PATIENT_NAME}' not found"
    
    def parsing_target_studies(self, studies: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """
        해당 환자의 검사(Studies) 리스트로부터 Study 메인 태그 정보 및 series UUIDs 추출

        Args:
            studies (List[str]): Study UUID 리스트

        Returns:
            Tuple[Dict[str, Any], List[str]]:
                - Study 메인 태그 정보 (StudyDate, StudyDescription, StudyInstanceUID 등)
                - Series UUID 리스트
        """
        for study in studies:
            studies_info = requests.get(f"{self.ORTHANC_URL}/studies/{study}").json()
            studies_tag_data: Dict[str, Any] = studies_info.get("MainDicomTags", {})
            series: List[str] = studies_info.get("Series", [])
        return studies_tag_data, series

    def parsing_target_series(self, series: List[str]) -> Tuple[List[str], str]:
        """
        Series UUIDs로 부터 Series 정보 추출 및 Instances UUID 추출

        Args:
            series (List[str]): Series UUID 리스트

        Returns:
            Tuple[List[str], str]:
                - Instances UUID 리스트
                - Modality (예: 'OT', 'CT', 'MR' 등)
        """
        for serial in series:
            series_info = requests.get(f"{self.ORTHANC_URL}/series/{serial}").json()
            instances_data: List[str] = series_info.get("Instances", [])
            modality: str = series_info.get("MainDicomTags", {}).get("Modality", "")
        return instances_data, modality
    
    def parsing_target_dicom_image(self, instances_data: List[str]) -> List[np.ndarray]:
        """
        해당 환자의 복부 X-Ray DICOM Image 추출

        Args:
            instances_data (List[str]): Instances UUID 리스트

        Returns:
            List[np.ndarray]: 각 Instance의 numpy 이미지 배열 (H, W, C)
        """
        instances_images: List[np.ndarray] = []
        for instance in instances_data:
            r = requests.get(f"{self.ORTHANC_URL}/instances/{instance}/numpy")
            image: np.ndarray = np.load(io.BytesIO(r.content))
        return image
