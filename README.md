# Kaggle Titanic - Machine Learning from Disaster
Link contest: https://www.kaggle.com/competitions/titanic
## 1. Xử lý dữ liệu đầu vào
Dữ liệu được làm sạch và chuẩn hóa thông qua class `DataProcess` áp dụng cho cả tập `train` và `test`:
* **Loại bỏ dữ liệu nhiễu:** Xóa các cột `PassengerId`, `Ticket` không mang ý nghĩa phân tích.
* **Xử lý dữ liệu khuyết thiếu (Missing Values):**
  * `Age`: Điền bằng độ tuổi trung bình (mean) của tập dữ liệu.
  * `Cabin`: Điền bằng giá trị phòng xuất hiện nhiều nhất (mode) dựa theo hạng vé (`Pclass`).
* **Feature Engineering (Tạo đặc trưng mới):**
  * `Title`: Rút trích danh xưng (Mr, Mrs, Miss...) từ cột `Name`. Gom các danh xưng hiếm (Lady, Don, Rev...) thành nhóm `Rare`.
  * `FamilySize`: Tính toán quy mô gia đình bằng cách cộng `SibSp`, `Parch` và bản thân hành khách (+1).
  * `IsAlone`: Biến nhị phân đánh dấu hành khách đi một mình (`FamilySize == 1`).
  * Xóa các cột thô `Name`, `SibSp`, `Parch` sau khi đã trích xuất xong đặc trưng.
* **Mã hóa (Categorical Encoding):** Áp dụng `OneHotEncoder` để chuyển hóa các cột chữ (`Title`, `Sex`, `Embarked`, `Cabin`) sang ma trận nhị phân để mô hình có thể tính toán.

## 2. Quá trình huấn luyện mô hình
* **Thuật toán sử dụng:** Random Forest Classifier.
* **Quy trình:** * Dữ liệu được chia tách thành ma trận đặc trưng `X_train` và nhãn `y_train` (cột `Survived`).
  * Huấn luyện mô hình phân loại trên tập `train`.
  * Đưa tập `X_test` đã qua tiền xử lý vào dự đoán.
  * Chuyển đổi mảng dự đoán thành Dataframe và xuất ra định dạng file `submission.csv` phục vụ cho nộp bài.

## 3. Tuning để tối ưu điểm số
Dự án được tối ưu hóa ở cả hai cấp độ: Dữ liệu và Siêu tham số mô hình.
* **Tối ưu hóa dữ liệu:** * Cải tiến hàm xử lý `Cabin` bằng phương thức `convertCabinName()`. Chỉ lấy ký tự đầu tiên của mã phòng (đại diện cho khu vực boong tàu - Deck) thay vì mã phòng chi tiết. Điều này giúp giảm thiểu sự phân mảnh dữ liệu.
* **Tối ưu hóa siêu tham số (Hyperparameter Tuning):**
  * Sử dụng thuật toán `GridSearchCV` kết hợp Cross Validation để tìm ra bộ tham số tối ưu nhất cho Random Forest.
  * Các tham số được tập trung tinh chỉnh bao gồm độ sâu của cây (`max_depth`) với các giá trị thử nghiệm: `[5, 10, 15, None]`.
