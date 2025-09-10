import torch
from torchvision.models import resnet18
from torchvision import transforms
import os

mapping = {
    0: "광선각화증",
    1: "기저세포암",
    2: "멜라닌세포모반",
    3: "보웬병",
    4: "비립종",
    5: "사마귀",
    6: "악성흑생종",
    7: "지루각화증",
    8: "편평세포암",
    9: "표피낭종",
    10: "피부섬유종",
    11: "피지샘증식증",
    12: "혈관종",
    13: "화농육아종",
    14: "흑색점"
}

# Image Classfication Resnet18 Model 
def load_model(model_path='../models/Skin_RESNET_18.pth', num_classes=15):
    model_path = os.path.join(os.path.dirname(__file__), model_path)
    model = resnet18(pretrained=False, num_classes=num_classes)
    model = model.to('cpu')

    # strict=False로 파라미터 일부 무시하며 로드
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Image PreProcessing (Pixel값 0 ~ 1 정규화 후 Torch Tensor 변환)
def preprocess(image_tensor):
    return torch.from_numpy(image_tensor).permute(0,3,1,2)/255.0

# Image Classification Predict
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output,dim=1)
        # 상위 5개 클래스 뽑기
        _, topk_indices = torch.topk(probs, k=5)
        topk_indices = topk_indices.squeeze(0)
        # 가장 높은 확률의 클래스
        pred_idx = int(torch.argmax(probs).item())
        pred_label = mapping[pred_idx]
        probs_dict = {mapping[i.item()]: float(probs[0][i.item()]) for i in topk_indices}
        return probs_dict, pred_label
