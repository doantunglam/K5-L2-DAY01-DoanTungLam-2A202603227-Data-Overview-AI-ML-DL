# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:** 3.13.18 / 2.11.0+cpu / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   -  Record này nói về toàn bộ bức ảnh chứ không phải về một vật thể riêng lẻ. Vì mô hình phân loại ảnh (yolo11n-cls.pt) được huấn luyện trên tập dữ liệu ImageNet-1K để gán một nhãn duy nhất (hoặc xếp hạng các nhãn phù hợp nhất) cho toàn bộ khung hình, chứ không xác định vị trí tọa độ của từng vật thể riêng lẻ.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Danh sách tên lớp (class list) không phải do mô hình tự nghĩ ra, mà đi kèm với taxonomy / checkpoint đã được huấn luyện từ trước. Mô hình AI khi dự đoán chỉ trả về các chỉ số lớp (class ID như 0, 1, 2...). Để chuyển từ chỉ số số học đó thành tên vật thể đọc được, mô hình phải tra cứu bảng ánh chỉ số - tên lớp (taxonomy) được lưu sẵn trong file checkpoint (.pt) của bộ dữ liệu đã học.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   - Cần ghi cả 3 thông tin để đảm bảo tính chính xác, không nhầm lẫn khi xử lý dữ liệu và huấn luyện mô hình:
   - class_id: Giúp máy tính và thuật toán xử lý nhanh, chính xác khi tính toán
   - class_name: Dành cho con người đọc và hiểu trực quan vật thể đó là gì
   - taxonomy_name: Xác định rõ ngữ cảnh và nguồn của danh mục nhãn
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   - Guideline cần quy định rõ các quy tắc ưu tiên sau để người gán nhãn chọn đúng một lớp duy nhất:
   - Vật thể chính: Ưu tiên đối tượng chiếm diện tích lớn nhất, ở trung tâm hoặc nằm ở tiêu điểm.
   - Thứ tự ưu tiên giữa các lớp: Liệt kê danh sách ưu tiên khi có nhiều vật thể xuất hiện cùng lúc.
   - Quy tắc xử lý ca mờ mập mờ: Hướng dẫn cụ thể khi các đối tượng có độ quan trọng ngang nhau.
6. Vì sao không được lấy `score` của model làm ground truth?
   - Score chỉ thể hiện độ tự tin chứ không đảm bảo độ chính xác, vì mô hình có thể tin 99% nhưng vẫn đoán sai hoàn toàn. Nếu lấy score của máy làm ground truth, mô hình sẽ lặp lại chính lỗi sai của nó thay vì được đánh giá và điều chỉnh khách quan bởi con người.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=car ; `score=0.863074 ; `bbox_xyxy`=[419.05, 264.24, 497.95, 339.85] ; `bbox_width`=78.9 ; `bbox_height`=75.61
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   - Hộp giới hạn này có góc trên-trái đặt tại tọa độ pixel (419.05, 264.24) và góc dưới-phải đặt tại tọa độ pixel (497.95, 339.85), với chiều rộng là 78.9 pixel và chiều cao là 75.61 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
   - Với ngưỡng 0.20, mô hình phát hiện được 17 vật thể (predictions).
   - Với ngưỡng 0.60, mô hình phát hiện được 6 vật thể (predictions).
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Khi hạ ngưỡng phát hiện để giữ lại nhiều hộp dự đoán hơn, người reviewer phải tốn nhiều thời gian và công sức hơn để thủ công tìm và xóa bỏ các hộp dự đoán sai lệch, đồng thời phải căn chỉnh lại kích thước, vị trí của các hộp có độ ôm khít kém và sửa lại các nhãn phân loại bị nhầm lẫn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   - Người gán nhãn phải vẽ hộp giới hạn ôm sát tối đa vào các cạnh biên ngoài cùng của vật thể nhưng tuyệt đối không được cắt lẹm vào bộ phận của vật thể. Trong trường hợp vật thể bị che khuất một phần hoặc bị khuất khỏi rìa ảnh, người gán nhãn vẫn vẽ hộp bao quanh toàn bộ phần nhìn thấy được của vật thể đó và không tự ý ước lượng vẽ thêm phần bị che khuất trừ khi có quy định cụ thể khác.
8. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   - Trên ảnh kitchen, chiếc lò nướng (oven) là vật thể bị che khuất một phần bởi bàn bếp phía trước. Trường hợp này nên được đưa vào quy tắc trong guideline thay vì đưa lên hỏi Lab Coach (escalation). Lý do là vì hiện tượng vật thể che khuất nhau hoặc bị cắt mép là lỗi xảy ra liên tục và phổ biến trong thực tế; việc thiết lập quy tắc rõ ràng trong guideline giúp toàn bộ đội ngũ tự động thống nhất cách vẽ ôm sát phần nhìn thấy mà không làm gián đoạn quy trình hoặc gây quá tải cho Lab Coach.



## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
   - So với hộp giới hạn (box) vốn chỉ cung cấp phạm vi hình chữ nhật bao quanh vật thể, đa giác (polygon) cung cấp ranh giới chính xác theo từng điểm bám sát viền ngoài thực tế của vật thể. Điều này giúp xác định chi tiết hình dạng hình học phức tạp, diện tích bề mặt thực tế và các góc cắt hay độ cong của vật thể mà không bị lẫn lộn các khoảng trống hoặc vật thể khác xung quanh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   - Trong bài toán phân đoạn thực thể, instance_id được dùng để phân biệt độc lập và định danh riêng cho từng đối tượng đơn lẻ trong cùng một bức ảnh, ngay cả khi chúng thuộc cùng một lớp. Câu nói trên có nghĩa là instance_id không dùng để phân loại nhóm vật thể (như class_id) và cũng không dùng để theo dõi liên tục một vật thể di chuyển qua chuỗi nhiều khung hình video khác nhau (như tracking_id), mà nó chỉ có giá trị phân định duy nhất cho từng thực thể xuất hiện trong phạm vi một bức ảnh tĩnh cụ thể.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   - Người gán nhãn phải chấm các điểm đa giác bám sát chính xác theo đường biên phân chia thực tế giữa vật thể và nền cảnh (sai số không quá 1 pixel), tuyệt đối không được vẽ lấn ra ngoài phần nền hoặc cắt lẹm vào cấu trúc bên trong của vật thể.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   - Trên ảnh kitchen, khu vực các chiếc bát (bowl) xếp chồng lên nhau và chạm sát nhau là ví dụ điển hình cho ranh giới mờ và chạm nhau. Trường hợp này nên được đưa vào quy tắc trong guideline thay vì escalation vì đây là tình huống thường gặp khi gán nhãn đồ dùng nhà bếp; việc quy định rõ cách phân tách ranh giới hoặc gán nhãn đè lên nhau trong guideline sẽ giúp annotator chủ động xử lý nhất quán trên quy mô lớn mà không cần hỏi lại Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh: `class_id`, `class_name`, `taxonomy_name` | Ảnh `traffic` có `cab` 0.510915 đứng đầu dù chủ yếu là xe buýt và ô tô; ảnh `kitchen` bị đoán `gong` 0.420191 | Xem cả ảnh, chọn một lớp trong taxonomy theo tiêu chí của guideline | Lớp có đúng taxonomy và đúng tiêu chí không; ca nhiều chủ đề có xử lý thống nhất không |
| Phát hiện vật thể | Mỗi vật một box `xyxy` (pixel) kèm lớp | Cánh tay cắt mép `person` 0.610931 tại [0.12, 263.18, 61.55, 310.78]; `oven` 0.6343 tại [489.09, 200.73, 617.81, 344.88] trông giống tủ có chậu kim loại hơn là lò; ảnh có bàn lớn nhưng ở ngưỡng 0.35 không có box `dining table` (chỉ xuất hiện ở 0.20) | Vẽ box ôm sát phần nhìn thấy cho từng vật và gán lớp | Có sót vật, box trùng, sai lớp hoặc box lệch không; hộp điểm thấp có đúng không |
| Instance segmentation | Mỗi đối tượng một polygon/mask kèm lớp và `instance_id` riêng | Bát `kitchen-006` và bàn `kitchen-005` chồng nhau khoảng 907 pixel; `kitchen-006` còn sát mép trái ảnh | Vẽ biên theo phần nhìn thấy, tách riêng từng đối tượng, xử lý chỗ chồng theo guideline | Biên có sát vật không; mask có lấn sang vật khác không; mỗi đối tượng có đúng một instance không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   - Quy tắc bảo vệ dữ liệu tuyệt đối của bài lab này là chỉ sử dụng ba bức ảnh COCO công khai có sẵn, không tải lên hay sử dụng bất kỳ hình ảnh cá nhân hoặc dữ liệu nội bộ nào của doanh nghiệp lên môi trường Colab.
3. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Lab Coach

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
