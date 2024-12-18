
import google.generativeai as genai
import os

# Thiết lập khóa API bằng cách sử dụng tên biến môi trường chính xác
genai.configure(api_key=os.environ["API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")
query = input("Nhap cau hoi ")
response = model.generate_content(query)
print(response.text)