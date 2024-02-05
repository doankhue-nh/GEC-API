import requests

# Địa chỉ và cổng của ứng dụng Flask
api_url = "http://0.0.0.0:3000/components/model"

# Dữ liệu JSON để gửi trong yêu cầu POST
data = {
    "model": "GECToR-Roberta",  # Chọn mô hình
    "text_input_list": ["Me and him are going to the store.", "His name am Khue"]
}

# Thực hiện yêu cầu POST đến API
response = requests.post(api_url, json=data)

# Kiểm tra phản hồi từ API
if response.status_code == 200:
    result = response.json()
    print(f"Model: {result['model']}")
    print("Text Outputs:")
    for i, output in enumerate(result['text_output_list'], 1):
        print(f"Output {i}: {output}")
else:
    print(f"Error: {response.status_code}, {response.text}")
