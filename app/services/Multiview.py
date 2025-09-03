import os
import re
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from collections import defaultdict
from appendicitis_network import create_model
from app.core.config import settings

class MultiViewService:
    def __init__(self):
        self.model = None
        self.device = torch.device('cpu')  # CPU만 사용
        self.max_views = 20  # Dataset과 동일하게 설정
        self.concept_names = [
            "Appendix visible", "Free fluid", "Irregular layers", "Target sign",
            "Tissue reaction", "Lymphadenitis", "Thick bowel wall",
            "Coprostasis", "Meteorism"
        ]
        self._load_model()
    def _load_model(self):
        """MVCBM 모델 로드"""
        try:
            config = {
                'model': 'MVCBM',
                'device': self.device,
                'aggregator': 'lstm',  # Dataset에서 사용하는 LSTM aggregator
                'num_concepts': 9,
                'num_classes': 2,
                'attention': False,
                'fusion': False,
                'num_ex_feat': 0,
                't_hidden_dim': 5,
                'encoder_arch': 'ResNet18',
                'model_directory': str(settings.MODEL_DIR)
            }
            self.model = create_model(config)
            self.model = self.model.to(self.device)
           # 모델 경로를 Path 객체로
            model_path = Path("../models/appendicitis_model.pth")
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.eval()
                print(f"Multiview MVCBM model loaded successfully")
                return self.model
            else:
                print(f"Warning: Model file not found at {model_path}")
        except Exception as e:
            print(f"Error loading multiview MVCBM model: {str(e)}")
            raise
    
    def create_multiview_batch(self, images_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """
        단일 환자의 여러 이미지를 multiview 배치로 생성
        images_tensor: [num_images, C, H, W] 형태
        반환:
            - images_tensor: [1, num_views, C, H, W]
            - mask: [1, num_views] (실제 이미지 True, 패딩 False)
            - file_names: 파일 이름 리스트 (패딩은 'padding')
        """
        num_images = images_tensor.size(0)
        
        # 패딩 처리
        if num_images < self.max_views:
            padding_image = torch.zeros_like(images_tensor[0])
            pad_count = self.max_views - num_images
            padding_tensor = padding_image.unsqueeze(0).repeat(pad_count, 1, 1, 1)
            images_tensor = torch.cat([images_tensor, padding_tensor], dim=0)
            file_names = [f"img_{i}" for i in range(num_images)] + ["padding"] * pad_count
        else:
            images_tensor = images_tensor[:self.max_views]
            file_names = [f"img_{i}" for i in range(self.max_views)]

        # 배치 차원 추가
        images_tensor = images_tensor.unsqueeze(0)  # [1, num_views, C, H, W]
        
        # 마스크 생성: 실제 이미지는 True, 패딩은 False
        mask = torch.tensor([name != "padding" for name in file_names], dtype=torch.bool).unsqueeze(0)

        return images_tensor, mask, file_names

    
    def predict_multiview_tensor(self,
                                   images_tensor: torch.Tensor,
                                   mask: torch.Tensor,
                                   return_attention: bool = False) -> Dict:
        """
        Multiview 이미지 텐서로 appendicitis 예측
        images_tensor: [batch_size, num_views, C, H, W]
        mask: [batch_size, num_views] (True=실제 이미지, False=패딩)
        """
        try:
            if self.model is None:
                raise Exception("Model not loaded")

            images_tensor = images_tensor.to(self.device)
            mask = mask.to(self.device)

            batch_size = images_tensor.size(0)
            results = {}

            for i in range(batch_size):
                try:
                    images_batch = images_tensor[i:i+1]  # [1, num_views, C, H, W]
                    mask_batch = mask[i:i+1]            # [1, num_views]
                    ex_feat = torch.empty((1, 0), device=self.device)

                    with torch.no_grad():
                        concepts_pred, target_pred_probs, target_pred_logits, attn_weights = \
                            self.model(images_batch, mask_batch, ex_feat)

                    concepts_scores = concepts_pred.cpu().numpy().flatten()
                    appendicitis_prob = target_pred_probs.cpu().numpy().flatten()[0]

                    concept_dict = {
                        name: float(score)
                        for name, score in zip(self.concept_names, concepts_scores)
                    }

                    result = {
                        'success': True,
                        'num_views': mask_batch.sum().item(),
                        'appendicitis_probability': float(appendicitis_prob),
                        'concept_scores': concept_dict,
                        'raw_concepts': concepts_scores.tolist(),
                        'target_logits': target_pred_logits.cpu().numpy().flatten().tolist()
                    }

                    if return_attention and attn_weights is not None:
                        result['attention_weights'] = attn_weights.cpu().numpy().tolist()

                    results[f'patient_{i}'] = result

                except Exception as e:
                    results[f'patient_{i}'] = {
                        'success': False,
                        'error': str(e)
                    }

            return {
                'success': True,
                'total_patients': batch_size,
                'results': results
            }

        except Exception as e:
            return
    
    async def predict_single_patient_multiview(self,
                                             patient_image_paths: List[str],
                                             return_attention: bool = False) -> Dict:
        """
        단일 환자의 multiview 이미지들로 예측 (환자 ID 추출 없이)
        """
        try:
            if self.model is None:
                raise Exception("Model not loaded")
            # Multiview 배치 생성
            images_tensor, mask, file_names = self.create_multiview_batch(patient_image_paths)
            images_tensor = images_tensor.to(self.device)
            mask = mask.to(self.device)
            # 추가 tabular features (없으므로 빈 텐서)
            ex_feat = torch.empty((1, 0), device=self.device)
            # 모델 추론
            with torch.no_grad():
                concepts_pred, target_pred_probs, target_pred_logits, attn_weights = \
                    self.model(images_tensor, mask, ex_feat)
            # 결과 처리
            concepts_scores = concepts_pred.cpu().numpy().flatten()
            appendicitis_prob = target_pred_probs.cpu().numpy().flatten()[0]
            # 개념별 점수 딕셔너리 생성
            concept_dict = {
                name: float(score)
                for name, score in zip(self.concept_names, concepts_scores)
            }
            result = {
                'success': True,
                'num_views': len([f for f in file_names if f != "padding.bmp"]),
                'appendicitis_probability': float(appendicitis_prob),
                'concept_scores': concept_dict,
                'raw_concepts': concepts_scores.tolist(),
                'target_logits': target_pred_logits.cpu().numpy().flatten().tolist(),
                'image_files': [f for f in file_names if f != "padding.bmp"]
            }
            # Attention weights 추가 (요청된 경우)
            if return_attention and attn_weights is not None:
                result['attention_weights'] = attn_weights.cpu().numpy().tolist()
            return result
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'num_views': 0,
                'appendicitis_probability': None,
                'concept_scores': None
            }
# 전역 인스턴스
multiview_service = MultiViewService()