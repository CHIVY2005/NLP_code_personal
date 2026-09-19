# Assignment 2: Neural Dependency Parser

## Tổng quan

Bài tập xây dựng một **transition-based dependency parser** bằng PyTorch. Chương trình nhận một câu, lần lượt chọn các phép chuyển trạng thái và tạo cây phụ thuộc cú pháp dưới dạng các cặp `(head, dependent)`.

Luồng xử lý chính:

```text
Câu đầu vào
    -> trạng thái (stack, buffer, dependencies)
    -> trích xuất 36 đặc trưng
    -> mạng neural dự đoán S / LA / RA
    -> cập nhật trạng thái
    -> lặp đến khi parse hoàn tất
    -> danh sách quan hệ phụ thuộc
```

## 1. `parser_transitions.py`

### `PartialParse.__init__`

Khởi tạo trạng thái parse:

- `stack = ["ROOT"]`: stack ban đầu chỉ có nút gốc.
- `buffer = list(sentence)`: chứa các từ chưa xử lý. Việc sao chép bảo đảm không sửa câu đầu vào.
- `dependencies = []`: nơi lưu các cạnh phụ thuộc đã tạo.

### `parse_step(transition)`

Thực hiện một trong ba phép chuyển:

- `S` (SHIFT): lấy từ đầu `buffer` đưa lên đỉnh `stack`.
- `LA` (LEFT-ARC): với hai phần tử cuối là `[..., dependent, head]`, thêm cạnh `(head, dependent)` rồi xóa `dependent`.
- `RA` (RIGHT-ARC): với hai phần tử cuối là `[..., head, dependent]`, thêm cạnh `(head, dependent)` rồi xóa `dependent`.

Ví dụ với `stack = [ROOT, run, fast]`, phép `RA` tạo cạnh `(run, fast)` và stack còn `[ROOT, run]`.

### `parse(transitions)`

Áp dụng tuần tự một danh sách transition bằng `parse_step`, sau đó trả về toàn bộ dependency đã tạo.

### `minibatch_parse(sentences, model, batch_size)`

Tạo một `PartialParse` cho mỗi câu rồi lặp như sau:

1. Lấy tối đa `batch_size` parse chưa hoàn tất.
2. Gọi `model.predict(...)` một lần cho cả minibatch.
3. Áp dụng transition tương ứng lên từng parse.
4. Loại các parse đã có buffer rỗng và stack chỉ còn một phần tử.
5. Trả dependency theo đúng thứ tự câu đầu vào.

Cách làm này giúp mạng neural dự đoán song song cho nhiều câu, hiệu quả hơn gọi model riêng cho từng câu. Hàm cũng kiểm tra `batch_size > 0` và số transition model trả về phải khớp số parse trong batch.

## 2. `parser_model.py`

`ParserModel` là mạng feed-forward dùng để phân loại transition tiếp theo.

### Các tham số

Với mặc định 36 đặc trưng, embedding 50 chiều, hidden size 200 và 3 lớp đầu ra:

- `embeddings`: ma trận embedding có thể học, kích thước `(vocab_size, 50)`.
- `embed_to_hidden_weight`: trọng số từ input đến hidden, kích thước `(36 * 50, 200)`.
- `embed_to_hidden_bias`: bias hidden, kích thước `(200,)`.
- `hidden_to_logits_weight`: trọng số hidden đến output, kích thước `(200, 3)`.
- `hidden_to_logits_bias`: bias output, kích thước `(3,)`.
- `dropout`: regularization, mặc định loại ngẫu nhiên 50% hidden unit khi train.

Các ma trận trọng số được khởi tạo bằng Xavier uniform để giữ độ lớn tín hiệu ổn định qua các lớp.

### `embedding_lookup(w)`

Input `w` có kích thước `(batch_size, n_features)` và chứa ID token. Hàm tra embedding bằng tensor indexing, sau đó nối embedding của 36 đặc trưng thành tensor:

```text
(batch_size, n_features, embed_size)
    -> (batch_size, n_features * embed_size)
```

Không dùng vòng lặp nên đáp ứng yêu cầu hiệu năng của đề.

### `forward(w)`

Phép lan truyền tiến là:

```text
x      = embedding_lookup(w)
hidden = ReLU(x @ W + b1)
hidden = Dropout(hidden)
logits = hidden @ U + b2
```

Hàm trả `logits`, không gọi softmax. `nn.CrossEntropyLoss` đã thực hiện log-softmax ổn định số học bên trong.

## 3. `run.py`

### `train(...)`

- Tạo optimizer Adam với learning rate mặc định `0.0005`.
- Dùng `nn.CrossEntropyLoss` cho bài toán phân loại ba lớp.
- Huấn luyện nhiều epoch và đánh giá trên dev set sau mỗi epoch.
- Chỉ lưu `state_dict` của model có dev UAS tốt nhất.

`best_dev_UAS` bắt đầu từ âm vô cực để epoch đầu tiên luôn có thể tạo checkpoint, kể cả trong một test nhỏ có UAS bằng 0.

### `train_for_epoch(...)`

Mỗi minibatch thực hiện đúng chu trình PyTorch:

```python
optimizer.zero_grad()
logits = parser.model(train_x)
loss = loss_func(logits, train_y)
loss.backward()
optimizer.step()
```

Nhãn one-hot được đổi sang class index bằng `argmax`. Loss trung bình được tính theo số mẫu thực tế, nên minibatch cuối nhỏ hơn không làm lệch thống kê.

Sau một epoch, code chuyển sang `model.eval()` để tắt dropout và tính UAS trên dev set.

## 4. `utils/parser_utils.py`

File tiện ích này đã được đề cung cấp, không phải phần cần cài đặt. Nó chịu trách nhiệm:

- đọc dữ liệu CoNLL;
- tạo từ điển token/POS;
- vector hóa dữ liệu;
- lấy 36 đặc trưng từ stack, buffer và các arc hiện tại;
- sinh transition oracle để tạo dữ liệu huấn luyện;
- giới hạn các transition hợp lệ;
- bọc model để dùng với `minibatch_parse`;
- tính UAS (Unlabeled Attachment Score).

UAS là tỷ lệ từ có **head dự đoán đúng**, không xét nhãn loại quan hệ.

## 5. Cách chạy kiểm tra

Chạy từ thư mục `a2/a2`:

```bash
python parser_transitions.py part_c
python parser_transitions.py part_d
python parser_model.py -e -f
python run.py -d
```

Ý nghĩa:

- `part_c`: kiểm tra SHIFT, LEFT-ARC, RIGHT-ARC và chuỗi parse.
- `part_d`: kiểm tra minibatch parsing.
- `-e -f`: kiểm tra embedding lookup và shape của forward pass.
- `run.py -d`: train trên tập rút gọn để debug nhanh.

Khi debug ổn, chạy toàn bộ dữ liệu:

```bash
python run.py
```

Theo đề, debug thường cần đạt train loss dưới `0.2` và dev UAS trên `65%`; full run thường cần loss dưới `0.08` và dev UAS trên `87%`. Kết quả có dao động do khởi tạo ngẫu nhiên và dropout.

## 6. Quan hệ giữa các file

```text
run.py
  -> load_and_preprocess_data() trong utils/parser_utils.py
  -> ParserModel trong parser_model.py
  -> train model
  -> Parser.parse()
       -> ModelWrapper.predict()
       -> minibatch_parse() trong parser_transitions.py
       -> dependency edges và UAS
```

Tóm lại, `parser_transitions.py` định nghĩa thuật toán parse, `parser_model.py` học cách chọn transition, `run.py` huấn luyện/đánh giá, còn `parser_utils.py` chuẩn bị dữ liệu và nối các phần với nhau.
