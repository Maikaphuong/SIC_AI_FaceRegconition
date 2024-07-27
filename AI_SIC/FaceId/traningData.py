import cv2
import os
import numpy as np
from PIL import Image

# Tạo ra một đối tượng nhận dạng khuôn mặt bằng thuật toán LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Đường dẫn đến thư mục chứa hình ảnh khuôn mặt, sử dụng raw string để tránh lỗi unicode
path = r'D:\Nam 2\LearnAI\AI_SIC\FaceId\dataSet'

# Hàm đọc các hình ảnh khuôn mặt từ thư mục
# Trả về Id của người trong ảnh và hình ảnh tương ứng
def getImagesAndLabels(path):
    # Sử dụng mô-đun os để lấy danh sách tất cả các tệp hình ảnh trong thư mục rồi lặp qua từng tệp để đọc dữ liệu hình ảnh
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] 
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.waitKey(1)
    return IDs, faces

# Hàm gọi hàm getImagesAndLabels lấy IDs và faces 
def trainData():
    Ids, faces = getImagesAndLabels(path)
    # Sử dụng phương thức train của recognizer
    recognizer.train(faces, np.array(Ids))
    # Lưu vào tệp trainningData.yml
    if not os.path.exists('recognizer'):
        os.makedirs('recognizer')
    recognizer.save('recognizer/trainningData.yml')
    print('Train success')

trainData()
cv2.destroyAllWindows()
