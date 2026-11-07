# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603641
- **Họ và tên:** [HỌ TÊN CỦA BẠN]
- **SEED** (in ở ô đầu notebook): [SỐ SEED]

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.857 | 0.824 | 0.283 | 0 |
| 2. BoW + MLP | 0.938 | 0.889 | 2.271 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.935 | 0.974 | 16.977 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.927 | 0.947 | 9.919 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.934 | 0.974 | 8.103 | 0 |
| 4a. LLM nhỏ zero-shot | không đo | 0.947 | 1215.141 | 0.006 |
| 4b. LLM lớn zero-shot | không đo | 0.974 | 966.897 | 0.056 |
| 4c. LLM nhỏ few-shot | không đo | 0.974 | 1003.156 | 0.018 |

Chênh F1 dưới ~0.02 trên test và dưới ~0.05 trên SUB 40 tin coi như hòa.

## 3. Dự đoán và thực tế

1. **Learning rate (thí nghiệm 1.B).** Em đoán lr = 50 làm loss dao động và không hội tụ. Thực tế loss vọt lên khoảng 2.6 ở vài bước đầu rồi tụt về 0.006, thấp hơn cả lr = 0.5. Em lệch vì không tính đến việc khi trọng số lớn, sigmoid bão hoà, error nhỏ dần nên bước cập nhật tự thu nhỏ. Loss train thấp ở đây không có nghĩa là mô hình tốt, vì mô hình quá tự tin và dễ overfit.
2. **Tin né từ khoá (thí nghiệm 3.B).** Em đoán 3b (Encoder + LogReg) đúng nhiều hơn BoW. Thực tế hòa 4/6: BoW sai tin "brand new phone at no cost", còn encoder sai tin "complimentary holiday", cả hai cùng bỏ sót tin "account rewards expire tonight". Em lệch vì mẫu quá nhỏ và encoder + LogReg tuyến tính chưa đủ tách các câu spam "mượt".
3. **50 case khó.** Em đoán LLM khó nhất ở nhóm prompt injection. Thực tế LLM đúng 100% nhóm này (4a đúng 98% tổng, 4d đúng 100%). Em lệch vì đã đánh giá thấp khả năng hiểu ý định của LLM, nhưng chỉ có 6 case nên chưa đủ để kết luận LLM an toàn trước injection.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất: F1 từ 0.824 (SUB) xuống 0.286 trên 50 case khó (accuracy 0.50). BoW + MLP tụt từ 0.889 xuống 0.400. Các model encoder tụt vừa phải (F1 0.612 đến 0.741), còn LLM gần như không tụt (4a 0.983, 4d 1.000). Điểm test thường lạc quan vì notebook chỉ bỏ tin trùng tuyệt đối, nên các spam mẫu chỉ khác số điện thoại hay mã thưởng vẫn nằm ở cả train lẫn test.

Nhóm khó nhất với BoW là spam giọng tự nhiên (0% và 10%), né từ khoá (17% và 0%) và thao túng/lừa đảo (12% và 38%). Với encoder là spam giọng tự nhiên (30–40%), thao túng với 3b/3c (25%) và nói về spam (50% với 3a/3b).

Hai case cụ thể:
- **[Case 1 — chép từ ô "Đọc lỗi" trong notebook]:** BoW sai vì chỉ cộng trọng số từng từ có trong từ điển 2.000 từ; tin không có từ khoá quen thuộc nên không có gì kéo điểm về spam.
- **[Case 2 — chép từ ô "Đọc lỗi"]:** encoder sai vì vector nghĩa của câu này gần với tin đời thường (hoặc câu cảnh báo về spam gần với spam thật), và bộ phân loại tuyến tính không tách được.

## 5. Thử thách cá nhân

Đề #2: giảm vocab còn 100 từ và tăng lên 5000 từ, train lại, báo F1 test, giải thích xu hướng.

| Kích thước vocab | Precision | Recall | F1 |
| --- | --- | --- | --- |
| 100 | 0.954 | 0.674 | 0.790 |
| 2000 | 1.000 | 0.750 | 0.857 |
| 5000 | 1.000 | 0.750 | 0.857 |

Khi tăng kích thước từ điển, F1 tăng rồi bão hoà. Với 100 từ, từ điển chỉ giữ các từ phổ biến nhất nên nhiều từ báo hiệu spam như "txt" hay "150p" không có mặt, mô hình bỏ sót nhiều spam hơn (recall 0.674). Từ 2000 lên 5000 thì kết quả giống hệt nhau vì 2000 từ đã chứa gần hết các từ có tín hiệu, các từ thêm vào quá hiếm, hầu như không xuất hiện trong tin test nên không đổi được dự đoán nào (lợi ích giảm dần). Recall giữ ở 0.75 cho thấy những spam còn lọt là loại không chứa từ khoá quen thuộc, đó là giới hạn của bag-of-words chứ không phải của kích thước từ điển.

Ghi chú: đề này yêu cầu không dùng AI; em có tham khảo đoạn code vòng lặp từ AI, đã chạy lại và tự đọc kết quả.

## 6. Phương án cho sản phẩm thật

Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày (~116 tin/giây): mọi cách tự train đều đạt (BoW + LogReg 0.28 ms, BoW + MLP 2.27 ms, 3c 8.1 ms, 3b 9.9 ms, 3a 17 ms). Ba cách LLM đều trượt: p95 khoảng 1.000–1.200 ms (gấp ~20 lần ngưỡng) và chỉ xử lý ~1 tin/giây khi gọi tuần tự.

Về chất lượng, BoW + MLP có F1 test cao nhất (0.938), nhưng chỉ còn 0.400 trên 50 case khó. Encoder + kNN còn 0.741 và p95 17 ms, nên nếu cần cả độ trễ lẫn độ bền thì em chọn encoder (3a hoặc 3c). LLM bền nhất (0.983–1.000) nhưng không đạt SLA.

Chi phí LLM theo bảng: LLM nhỏ $0.006/1k tin, tức khoảng $60/ngày (~$1.800/tháng) cho 10 triệu tin; LLM lớn $0.056/1k tin, khoảng $560/ngày (~$16.800/tháng). Các cách tự train gần như miễn phí khi chạy trên máy của mình.

Đề xuất router nhiều tầng: BoW + MLP xử lý toàn bộ tin, tin có p ≤ 0.3 hoặc p ≥ 0.7 thì quyết định luôn; tin lưỡng lự chuyển lên encoder + LogReg; nếu vẫn không chắc thì gửi LLM theo luồng bất đồng bộ để không làm trễ p95. Nếu chỉ ~5% tin đến LLM thì chi phí giảm từ ~$60 xuống ~$3/ngày. Đánh đổi: hệ thống phức tạp hơn, cần kiểm tra lại ngưỡng trên dữ liệu thật, và phần tin khó vẫn kém bền hơn dùng LLM cho tất cả.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| 1 | Nhờ chỉ lỗi và giải thích `vectorize` (TODO 1.1) | Chỉ ra thiếu kiểm tra `if w in vocab` (KeyError với từ lạ) và đưa bản sửa | Tự viết bản đầu, sửa theo, chạy ô kiểm tra ra  |
| 2–3 | Nhờ nhận xét code `build_mlp`, `train_mlp`, `cosine_similarity` và gợi ý cách làm `top_k_similar`, `knn_predict` | Xác nhận code đúng; giải thích hướng chuẩn hoá + nhân ma trận, chỉ lỗi `argmax` phải là `argsort` | Tự viết code, sửa `argmax` thành `argsort`, chạy lại ô kiểm tra |
| 4 | Nhờ viết `llm_classify_batch` | Vòng lặp gọi API qua `retry_with_backoff`, đếm `n_invalid`, ước tính token và USD | Kiểm tra chỉ chạy trên 40 tin SUB, `None` tính vào `n_invalid` |
| 4 | Debug F1 = 0, `invalid=40` | Hướng dẫn in phản hồi thô; phát hiện giới hạn 3 token làm câu trả lời rỗng | Tự chạy kiểm tra (3 token rỗng, 20 token ra "spam"), tăng `max_tokens` OpenAI lên 10, chạy lại thì invalid = 0 |

**Điều đã học:** Em hiểu ra rằng accuracy dễ đánh lừa với dữ liệu lệch lớp (đoán toàn ham đã được 87%), nên phải dùng precision, recall, F1. Điểm F1 cao trên test thường chưa chắc bền: trên 50 case khó các model nhỏ tụt rất mạnh, còn LLM thì bền nhưng quá chậm và tốn tiền, nên chọn phương án phải dựa trên số đo (F1, p95, USD) chứ không dựa trên cảm giác. Em cũng hiểu vì sao reasoning model cần `reasoning_effort="none"` và giới hạn token đủ lớn, nếu không câu trả lời sẽ rỗng.
