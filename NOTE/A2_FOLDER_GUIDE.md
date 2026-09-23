# Hướng dẫn toàn bộ thư mục `a2`: Neural Dependency Parser với PyTorch

Tài liệu này giải thích nội dung trong thư mục `a2` theo cùng phong cách với
`PYTORCH_NOTEBOOK_EXPLANATION.md`: đi từ ý tưởng, cú pháp Python/PyTorch, kích
thước tensor, chức năng từng class/hàm, đến luồng chạy hoàn chỉnh.

Phần chính của bài tập là một **transition-based neural dependency parser**:
mạng nơ-ron quan sát trạng thái parse hiện tại rồi chọn một trong ba thao tác
`SHIFT`, `LEFT-ARC`, `RIGHT-ARC` để dần xây dựng cây phụ thuộc của câu.

---

## 1. Cấu trúc thư mục

```text
a2/
├── A2_FOLDER_GUIDE.md              # tài liệu đang đọc
├── easy_NLP/
│   └── hw1.py                      # ví dụ sentiment dựa trên từ điển
├── __MACOSX/                       # metadata sinh ra khi giải nén trên macOS
└── a2/
    ├── README.txt                  # hướng dẫn tạo môi trường
    ├── Assignment2_Explanation.md  # bản tóm tắt ngắn
    ├── parser_transitions.py       # SHIFT/LA/RA và minibatch parsing
    ├── parser_model.py             # mạng feed-forward bằng PyTorch
    ├── run.py                      # train, validation, test, checkpoint
    ├── local_env.yml               # môi trường Conda
    ├── collect_submission.sh       # đóng gói bài nộp
    ├── data/
    │   ├── train.conll
    │   ├── dev.conll
    │   ├── test.conll
    │   ├── *.gold.conll
    │   └── en-cw.txt               # pretrained word embeddings
    └── utils/
        ├── parser_utils.py         # đọc dữ liệu, feature, oracle, UAS
        └── general_utils.py        # chia minibatch
```

Thư mục `__MACOSX` không thuộc logic chương trình. Đây là metadata do macOS thêm
vào file ZIP và có thể bỏ qua khi học code.

---

## 2. Dependency parsing là gì?

Dependency parsing tìm quan hệ phụ thuộc giữa các từ. Mỗi quan hệ là một cạnh:

```text
(head, dependent)
```

Ví dụ câu:

```text
I love NLP
```

có thể có các cạnh:

```text
(ROOT, love)
(love, I)
(love, NLP)
```

Cây trực quan:

```text
       ROOT
         │
        love
       /    \
      I     NLP
```

- `love` là từ trung tâm của câu và phụ thuộc vào `ROOT`.
- `I` và `NLP` phụ thuộc vào `love`.
- Mỗi từ, trừ `ROOT`, cần có đúng một head.

Bài tập đánh giá bằng **UAS — Unlabeled Attachment Score**:

\[
UAS = \frac{\text{số token có head dự đoán đúng}}
           {\text{tổng số token được đánh giá}}
\]

Từ “Unlabeled” nghĩa là chỉ kiểm tra head, không kiểm tra tên loại quan hệ như
`nsubj`, `obj` hay `det`.

---

## 3. Transition-based parsing

Parser không dự đoán toàn bộ cây trong một lần. Nó duy trì ba thành phần:

```python
stack = ["ROOT"]
buffer = list(sentence)
dependencies = []
```

- `stack`: các từ đang được xem xét;
- `buffer`: các từ chưa được đưa vào stack;
- `dependencies`: các cạnh đã tạo.

Mỗi bước parser chọn một transition.

### 3.1. `SHIFT` — ký hiệu `S`

Lấy phần tử đầu buffer và đưa lên đỉnh stack:

```text
stack:  [ROOT, I]
buffer: [love, NLP]

S

stack:  [ROOT, I, love]
buffer: [NLP]
```

Cú pháp trong code:

```python
self.stack.append(self.buffer.pop(0))
```

- `pop(0)` lấy và xóa phần tử đầu danh sách.
- `append(...)` thêm phần tử vào cuối stack.
- Trong bài tập nhỏ cách này dễ hiểu; với dữ liệu rất lớn, `collections.deque`
  có thao tác lấy đầu hiệu quả hơn list.

### 3.2. `LEFT-ARC` — ký hiệu `LA`

Với hai phần tử trên cùng:

```text
[..., dependent, head]
```

tạo `(head, dependent)` và xóa dependent:

```text
stack: [ROOT, I, love]

LA

dependency: (love, I)
stack:      [ROOT, love]
```

Code:

```python
head = self.stack[-1]
dependent = self.stack[-2]
self.dependencies.append((head, dependent))
self.stack.pop(-2)
```

Chỉ số âm trong Python:

- `stack[-1]`: phần tử cuối;
- `stack[-2]`: phần tử kế cuối.

### 3.3. `RIGHT-ARC` — ký hiệu `RA`

Với:

```text
[..., head, dependent]
```

tạo `(head, dependent)` và xóa dependent trên đỉnh stack:

```text
stack: [ROOT, love, NLP]

RA

dependency: (love, NLP)
stack:      [ROOT, love]
```

Code:

```python
head = self.stack[-2]
dependent = self.stack[-1]
self.dependencies.append((head, dependent))
self.stack.pop()
```

---

## 4. Mô phỏng parse hoàn chỉnh bằng tay

Với câu `I love NLP`, chuỗi transition có thể là:

```python
transitions = ["S", "S", "LA", "S", "RA", "RA"]
```

| Bước | Transition | Stack | Buffer | Dependency mới |
|---:|---|---|---|---|
| 0 | — | `[ROOT]` | `[I, love, NLP]` | — |
| 1 | `S` | `[ROOT, I]` | `[love, NLP]` | — |
| 2 | `S` | `[ROOT, I, love]` | `[NLP]` | — |
| 3 | `LA` | `[ROOT, love]` | `[NLP]` | `(love, I)` |
| 4 | `S` | `[ROOT, love, NLP]` | `[]` | — |
| 5 | `RA` | `[ROOT, love]` | `[]` | `(love, NLP)` |
| 6 | `RA` | `[ROOT]` | `[]` | `(ROOT, love)` |

Parser hoàn tất khi:

```python
not partial_parse.buffer and len(partial_parse.stack) == 1
```

tức buffer rỗng và stack chỉ còn `ROOT`.

---

## 5. `parser_transitions.py`

File này cài đặt thuật toán thay đổi trạng thái parse. Nó chưa dùng PyTorch.

### 5.1. Class `PartialParse`

```python
class PartialParse:
    def __init__(self, sentence):
        self.sentence = sentence
        self.stack = ["ROOT"]
        self.buffer = list(sentence)
        self.dependencies = []
```

Tại sao dùng `list(sentence)` thay vì gán trực tiếp?

```python
self.buffer = sentence       # hai biến trỏ cùng một list
self.buffer = list(sentence) # tạo bản sao nông
```

Parser sẽ xóa phần tử khỏi `buffer`. Tạo bản sao giúp câu đầu vào của người gọi
không bị thay đổi.

### 5.2. `parse_step()`

Phiên bản rút gọn:

```python
def parse_step(self, transition):
    if transition == "S":
        self.stack.append(self.buffer.pop(0))

    elif transition == "LA":
        head = self.stack[-1]
        dependent = self.stack[-2]
        self.dependencies.append((head, dependent))
        self.stack.pop(-2)

    elif transition == "RA":
        head = self.stack[-2]
        dependent = self.stack[-1]
        self.dependencies.append((head, dependent))
        self.stack.pop()

    else:
        raise ValueError("Transition must be S, LA, or RA")
```

Đề bài đảm bảo transition hợp lệ. Trong ứng dụng thực tế nên kiểm tra thêm:

- `S` chỉ hợp lệ khi buffer còn phần tử;
- `LA` và `RA` cần ít nhất hai phần tử trong stack;
- `ROOT` không được trở thành dependent của `LA`.

### 5.3. `parse()`

```python
def parse(self, transitions):
    for transition in transitions:
        self.parse_step(transition)
    return self.dependencies
```

Hàm này chỉ áp dụng một chuỗi transition có sẵn. Nó không tự quyết định
transition. Quyết định đó sẽ đến từ neural network hoặc oracle.

### 5.4. `minibatch_parse()`

Ý tưởng quan trọng: không gọi model riêng cho từng câu. Ta gom nhiều trạng thái
đang parse thành một batch:

```python
def minibatch_parse(sentences, model, batch_size):
    partial_parses = [PartialParse(sentence) for sentence in sentences]

    unfinished_parses = [
        parse for parse in partial_parses
        if parse.buffer or len(parse.stack) > 1
    ]

    while unfinished_parses:
        minibatch = unfinished_parses[:batch_size]
        transitions = model.predict(minibatch)

        for partial_parse, transition in zip(minibatch, transitions):
            partial_parse.parse_step(transition)

        unfinished_parses = [
            parse for parse in unfinished_parses
            if parse.buffer or len(parse.stack) > 1
        ]

    return [parse.dependencies for parse in partial_parses]
```

Luồng của một vòng lặp:

```text
các parse chưa xong
    -> lấy tối đa batch_size parse
    -> model.predict(minibatch)
    -> mỗi parse nhận một transition
    -> cập nhật stack/buffer/dependencies
    -> bỏ các parse đã hoàn tất
```

`partial_parses` ban đầu vẫn giữ đúng thứ tự câu. Vì vậy kết quả cuối cũng đúng
thứ tự, dù các câu hoàn thành ở thời điểm khác nhau.

### 5.5. `DummyModel` và test

`DummyModel` không học. Nó tạo transition bằng quy tắc cố định để kiểm tra thuật
toán batching độc lập với mạng nơ-ron:

```python
class DummyModel:
    def predict(self, partial_parses):
        return [
            "RA" if len(parse.buffer) == 0 else "S"
            for parse in partial_parses
        ]
```

Đây là một kỹ thuật thiết kế tốt: dùng fake/dummy dependency để test một thành
phần mà không cần train model thật.

---

## 6. Dữ liệu CoNLL trong `data/`

Các file `.conll` lưu mỗi token trên một dòng với các cột phân cách bằng tab.
Những trường quan trọng với bài tập:

| Cột | Nội dung |
|---:|---|
| 1 | chỉ số token trong câu |
| 2 | từ |
| 5 | POS tag |
| 7 | chỉ số head |
| 8 | dependency label |

Một dòng có thể được đọc bằng:

```python
columns = line.strip().split("\t")
word = columns[1]
pos = columns[4]
head = int(columns[6])
label = columns[7]
```

Câu mới bắt đầu sau một dòng trống. `read_conll()` tích lũy các trường rồi tạo:

```python
example = {
    "word": ["I", "love", "NLP"],
    "pos": ["PRP", "VBP", "NNP"],
    "head": [2, 0, 2],
    "label": ["nsubj", "root", "obj"],
}
```

Các file có `.gold.conll` chứa đáp án chuẩn để đánh giá. `en-cw.txt` chứa word
embedding 50 chiều đã được huấn luyện trước.

---

## 7. `utils/parser_utils.py`: chuẩn bị dữ liệu

Đây là file nối dữ liệu ngôn ngữ với mạng PyTorch.

### 7.1. Các token đặc biệt

```python
P_PREFIX = "<p>:"
L_PREFIX = "<l>:"
UNK = "<UNK>"
NULL = "<NULL>"
ROOT = "<ROOT>"
```

- `P_PREFIX`: phân biệt POS tag với từ có cùng chuỗi ký tự;
- `L_PREFIX`: phân biệt dependency label;
- `UNK`: token ngoài vocabulary;
- `NULL`: vị trí feature không tồn tại;
- `ROOT`: nút gốc.

Ví dụ `NN` là POS tag được lưu dưới key `<p>:NN`, tránh trùng với từ `NN`.

### 7.2. `Config`

```python
class Config:
    language = "english"
    with_punct = True
    unlabeled = True
    lowercase = True
    use_pos = True
    use_dep = False
```

Với cấu hình hiện tại:

- parser không dự đoán nhãn quan hệ, chỉ chọn `L`, `R`, `S`;
- dùng word feature và POS feature;
- không dùng dependency-label feature;
- tổng số feature là `18 + 18 = 36`.

### 7.3. Tạo vocabulary trong `Parser.__init__()`

Parser tạo ánh xạ:

```python
self.tok2id = {
    "<l>:root": 0,
    "<p>:NN": 1,
    "the": 2,
    # ...
}

self.id2tok = {value: key for key, value in self.tok2id.items()}
```

Mỗi từ/POS/label được đổi thành một integer ID. Neural network không nhận string
trực tiếp mà nhận các ID này.

### 7.4. `vectorize()`

```python
def vectorize(self, examples):
    vectorized = []

    for example in examples:
        words = [self.ROOT] + [
            self.tok2id.get(word, self.UNK)
            for word in example["word"]
        ]

        # POS, head và label được xử lý tương tự
        vectorized.append({
            "word": words,
            "pos": pos,
            "head": head,
            "label": label,
        })

    return vectorized
```

Phần tử `ROOT` được thêm ở index 0. Vì vậy index của token trong parser khớp với
quy ước head trong dữ liệu dependency.

### 7.5. 36 feature trong `extract_features()`

Với cấu hình `use_pos=True`, `use_dep=False`, model nhận:

```text
18 word features + 18 POS features = 36 feature IDs
```

18 word features gồm:

1. 3 từ trên cùng của stack;
2. 3 từ đầu buffer;
3. với mỗi từ trong 2 từ trên cùng của stack:
   - left child thứ nhất;
   - right child thứ nhất;
   - left child thứ hai;
   - right child thứ hai;
   - left-left grandchild;
   - right-right grandchild.

Tổng:

```text
3 + 3 + 2 * 6 = 18
```

Lặp lại đúng cấu trúc đó cho POS:

```text
18 word + 18 POS = 36
```

Nếu vị trí không tồn tại, ID `NULL` hoặc `P_NULL` được dùng:

```python
features = [self.NULL] * (3 - len(stack))
features += [example["word"][index] for index in stack[-3:]]
```

Ý nghĩa của phép nhân list:

```python
[self.NULL] * 2
```

tạo `[self.NULL, self.NULL]`.

Hàm kết thúc bằng kiểm tra rất hữu ích:

```python
assert len(features) == self.n_features
```

Nếu logic feature sai, chương trình dừng ngay tại nguồn thay vì để lỗi shape khó
hiểu xuất hiện trong mạng.

### 7.6. Oracle trong `get_oracle()`

Trong lúc train, ta biết cây đúng. Oracle nhìn cây đúng để chọn transition đúng
cho trạng thái hiện tại:

```text
nếu phần tử kế cuối stack phụ thuộc phần tử cuối -> LEFT-ARC
nếu phần tử cuối phụ thuộc phần tử kế cuối
   và không còn dependent của nó trong buffer -> RIGHT-ARC
nếu còn token trong buffer -> SHIFT
ngược lại -> không có transition hợp lệ
```

Tại sao RIGHT-ARC phải chờ tất cả dependent trong buffer?

Sau RIGHT-ARC, dependent trên đỉnh bị xóa khỏi stack. Nếu nó còn một child chưa
được SHIFT từ buffer, ta sẽ không thể nối child đó vào nó nữa.

### 7.7. `create_instances()`

Hàm chạy oracle trên từng câu để tạo dữ liệu supervised:

```python
training_instance = (
    features,      # 36 IDs mô tả trạng thái hiện tại
    legal_labels,  # transition nào hợp lệ
    gold_t,        # transition đúng
)
```

Mỗi câu `n` từ cần khoảng `2n` transition:

- `n` lần SHIFT để đưa từ vào stack;
- `n` lần ARC để tạo head cho mỗi từ và thu stack về `ROOT`.

### 7.8. `legal_labels()`

```python
def legal_labels(self, stack, buffer):
    left_arc = [1] if len(stack) > 2 else [0]
    right_arc = [1] if len(stack) >= 2 else [0]
    shift = [1] if len(buffer) > 0 else [0]
    return left_arc + right_arc + shift
```

Với unlabeled parser, thứ tự class là:

```text
0 -> LEFT-ARC
1 -> RIGHT-ARC
2 -> SHIFT
```

### 7.9. `load_and_preprocess_data()`

Toàn bộ pipeline dữ liệu:

```text
read_conll()
    -> train/dev/test dạng string
    -> Parser(train_set) tạo vocabulary
    -> đọc en-cw.txt
    -> tạo embedding matrix
    -> vectorize()
    -> create_instances(train_set)
    -> trả parser, embeddings, train examples, dev, test
```

Embedding matrix ban đầu:

```python
embeddings_matrix = np.asarray(
    np.random.normal(0, 0.9, (parser.n_tokens, 50)),
    dtype="float32",
)
```

Nếu token có trong `en-cw.txt`, hàng ngẫu nhiên được thay bằng pretrained vector.
Token không có pretrained embedding vẫn giữ vector ngẫu nhiên và sẽ được học.

---

## 8. `utils/general_utils.py`: tạo minibatch

`get_minibatches()` tạo một generator, trả từng batch mà không cần tạo tất cả
batch cùng lúc:

```python
def get_minibatches(data, minibatch_size, shuffle=True):
    indices = np.arange(data_size)

    if shuffle:
        np.random.shuffle(indices)

    for start in np.arange(0, data_size, minibatch_size):
        batch_indices = indices[start:start + minibatch_size]
        yield data[batch_indices]
```

`yield` khác `return`: hàm tạm dừng và tiếp tục từ vị trí cũ ở lần lặp kế tiếp.

Sử dụng:

```python
for x_batch, y_batch in get_minibatches([x, y], minibatch_size=1024):
    # train một batch
    pass
```

Shuffle giúp model không luôn nhìn dữ liệu theo cùng thứ tự ở mọi epoch.

---

## 9. `parser_model.py`: mạng nơ-ron PyTorch

### 9.1. Kiến trúc

Mạng nhận 36 token/POS IDs, tra embedding, nối các vector rồi phân loại thành ba
transition:

```text
feature IDs
    (B, 36)
        │ embedding lookup
        ▼
embeddings
    (B, 36, 50)
        │ flatten
        ▼
input vector
    (B, 1800)
        │ Linear + ReLU
        ▼
hidden
    (B, 200)
        │ Dropout
        ▼
hidden after dropout
    (B, 200)
        │ Linear
        ▼
logits
    (B, 3)
```

Trong đó `B` là batch size.

### 9.2. Vì sao kế thừa `nn.Module`?

```python
class ParserModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
```

`nn.Module` cung cấp:

- đăng ký và truy xuất parameters;
- `.train()` và `.eval()`;
- `.state_dict()` để lưu model;
- `.to(device)` để chuyển CPU/GPU;
- cơ chế gọi `model(input)` đến `forward()`.

### 9.3. Embedding là `nn.Parameter`

```python
self.embeddings = nn.Parameter(
    torch.as_tensor(embeddings, dtype=torch.float32).clone()
)
```

Giải thích:

- `torch.as_tensor(...)`: đổi NumPy array thành tensor;
- `dtype=torch.float32`: kiểu dữ liệu chuẩn cho trọng số;
- `.clone()`: tạo vùng nhớ riêng;
- `nn.Parameter(...)`: đăng ký embedding là trọng số có thể học.

Cách tương đương, quen thuộc hơn:

```python
self.embedding = nn.Embedding.from_pretrained(
    torch.tensor(embeddings, dtype=torch.float32),
    freeze=False,
)
```

### 9.4. Tự tạo các ma trận trọng số

Code không dùng `nn.Linear`; bài tập yêu cầu hiểu phép toán bên trong:

```python
self.embed_to_hidden_weight = nn.Parameter(
    torch.empty(n_features * embed_size, hidden_size)
)
self.embed_to_hidden_bias = nn.Parameter(torch.empty(hidden_size))

self.hidden_to_logits_weight = nn.Parameter(
    torch.empty(hidden_size, n_classes)
)
self.hidden_to_logits_bias = nn.Parameter(torch.empty(n_classes))
```

Với giá trị mặc định:

| Parameter | Shape |
|---|---|
| `embeddings` | `(vocab_size, 50)` |
| `embed_to_hidden_weight` | `(1800, 200)` |
| `embed_to_hidden_bias` | `(200,)` |
| `hidden_to_logits_weight` | `(200, 3)` |
| `hidden_to_logits_bias` | `(3,)` |

### 9.5. Xavier initialization

```python
nn.init.xavier_uniform_(self.embed_to_hidden_weight)
nn.init.xavier_uniform_(self.hidden_to_logits_weight)
```

Xavier initialization chọn miền giá trị dựa trên số input/output để tín hiệu và
gradient không tăng hoặc giảm quá nhanh qua layer.

Dấu `_` cuối `xavier_uniform_` cho biết hàm sửa tensor tại chỗ.

Bias được khởi tạo bằng:

```python
nn.init.uniform_(self.embed_to_hidden_bias)
```

Trong nhiều model hiện đại, bias cũng thường được khởi tạo bằng 0:

```python
nn.init.zeros_(self.embed_to_hidden_bias)
```

### 9.6. `embedding_lookup()`

```python
def embedding_lookup(self, w):
    x = self.embeddings[w]
    x = x.reshape(w.shape[0], self.n_features * self.embed_size)
    return x
```

Nếu:

```text
w.shape = (B, 36)
embeddings.shape = (V, 50)
```

advanced indexing tạo:

```text
self.embeddings[w].shape = (B, 36, 50)
```

Sau `reshape`:

```text
x.shape = (B, 36 * 50) = (B, 1800)
```

Không cần vòng lặp Python. Đây là vectorization: PyTorch tra toàn bộ embedding
trong một phép toán.

### 9.7. `forward()` từng dòng

```python
def forward(self, w):
    x = self.embedding_lookup(w)

    hidden = F.relu(
        x.matmul(self.embed_to_hidden_weight)
        + self.embed_to_hidden_bias
    )

    hidden = self.dropout(hidden)

    logits = (
        hidden.matmul(self.hidden_to_logits_weight)
        + self.hidden_to_logits_bias
    )

    return logits
```

Theo dõi shape:

```text
x                          (B, 1800)
embed_to_hidden_weight     (1800, 200)
x @ weight                 (B, 200)
+ bias (broadcast)         (B, 200)
ReLU                       (B, 200)
Dropout                    (B, 200)
hidden_to_logits_weight    (200, 3)
hidden @ weight            (B, 3)
+ output bias              (B, 3)
```

### 9.8. Broadcasting của bias

Phép tính:

```python
x.matmul(weight) + bias
```

cộng tensor `(B, 200)` với `(200,)`. PyTorch tự lặp bias cho mọi hàng trong
batch. Đây là broadcasting, không cần tự tạo bias shape `(B, 200)`.

### 9.9. ReLU

```python
hidden = F.relu(hidden)
```

Áp dụng theo từng phần tử:

\[
ReLU(x) = \max(0, x)
\]

ReLU thêm tính phi tuyến. Nếu chỉ xếp nhiều phép Linear mà không có activation,
toàn mạng vẫn tương đương một phép Linear duy nhất.

### 9.10. Dropout

```python
self.dropout = nn.Dropout(p=0.5)
hidden = self.dropout(hidden)
```

Khi train, dropout ngẫu nhiên đặt một phần hidden units về 0 để giảm overfitting.
Khi eval, dropout tự tắt:

```python
model.train()  # dropout hoạt động
model.eval()   # dropout tắt
```

### 9.11. Tại sao không Softmax trong `forward()`?

Model trả logits:

```python
return logits
```

Training dùng:

```python
loss_func = nn.CrossEntropyLoss()
loss = loss_func(logits, labels)
```

`CrossEntropyLoss` đã kết hợp LogSoftmax và negative log-likelihood theo cách ổn
định số học. Thêm Softmax trước loss là thừa và có thể làm gradient kém ổn định.

Khi cần xác suất để hiển thị:

```python
probabilities = torch.softmax(logits, dim=1)
predictions = logits.argmax(dim=1)
```

### 9.12. Phiên bản dùng các layer dựng sẵn

Model hiện tại cố ý tự tạo weight. Cùng kiến trúc có thể viết ngắn hơn:

```python
class ParserModelWithLayers(nn.Module):
    def __init__(self, embeddings, n_features=36,
                 hidden_size=200, n_classes=3, dropout_prob=0.5):
        super().__init__()

        embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(
            embedding_tensor,
            freeze=False,
        )
        embed_size = embedding_tensor.size(1)

        self.network = nn.Sequential(
            nn.Linear(n_features * embed_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, feature_ids):
        # (B, 36) -> (B, 36, 50)
        embedded = self.embedding(feature_ids)

        # (B, 36, 50) -> (B, 1800)
        flattened = embedded.flatten(start_dim=1)

        # (B, 1800) -> (B, 3)
        return self.network(flattened)
```

Phiên bản này phù hợp cho dự án thông thường; bản trong bài tập giúp hiểu rõ
`nn.Linear` thực chất là `x @ W + b`.

---

## 10. `run.py`: huấn luyện và đánh giá

### 10.1. Khởi tạo dữ liệu và model

```python
parser, embeddings, train_data, dev_data, test_data = \
    load_and_preprocess_data(debug)

model = ParserModel(embeddings)
parser.model = model
```

Ở đây có hai object khác nhau:

- `parser`: quản lý feature, oracle, parsing và UAS;
- `parser.model`: neural network dự đoán transition.

### 10.2. Optimizer và loss

```python
optimizer = optim.Adam(parser.model.parameters(), lr=0.0005)
loss_func = nn.CrossEntropyLoss()
```

`parser.model.parameters()` trả mọi `nn.Parameter` đã đăng ký, gồm embedding,
hai weight matrix và hai bias vector.

Adam giữ thêm moving averages của gradient để điều chỉnh bước cập nhật cho từng
parameter.

### 10.3. Một training step

```python
optimizer.zero_grad()

train_x = torch.from_numpy(train_x).long()
train_y = torch.from_numpy(train_y.argmax(axis=1)).long()

logits = parser.model(train_x)
loss = loss_func(logits, train_y)

loss.backward()
optimizer.step()
```

Thứ tự chuẩn:

```text
zero_grad -> forward -> loss -> backward -> step
```

Giải thích nhãn:

```python
train_y.argmax(axis=1)
```

đổi one-hot label:

```text
[0, 0, 1] -> 2 -> SHIFT
[1, 0, 0] -> 0 -> LEFT-ARC
[0, 1, 0] -> 1 -> RIGHT-ARC
```

`CrossEntropyLoss` yêu cầu label là class index kiểu `torch.long`, không phải
one-hot float.

### 10.4. `AverageMeter`

```python
loss_meter.update(loss.item(), train_x.shape[0])
```

`CrossEntropyLoss` mặc định trả mean loss của batch. Nhân theo kích thước batch
trong `AverageMeter` giúp minibatch cuối nhỏ hơn không có trọng số ngang với một
batch đầy đủ.

Công thức:

```python
self.sum += batch_loss * batch_size
self.count += batch_size
self.avg = self.sum / self.count
```

### 10.5. Validation và checkpoint

```python
parser.model.eval()
dev_UAS, _ = parser.parse(dev_data)

if dev_UAS > best_dev_UAS:
    best_dev_UAS = dev_UAS
    torch.save(parser.model.state_dict(), output_path)
```

Chỉ lưu model tốt nhất trên dev set, không nhất thiết model ở epoch cuối.

`state_dict()` là dictionary tên parameter → tensor:

```python
for name, tensor in parser.model.state_dict().items():
    print(name, tensor.shape)
```

Tải lại:

```python
parser.model.load_state_dict(torch.load(output_path))
parser.model.eval()
```

Với code mới hơn và khi chạy trên CPU, có thể ghi rõ:

```python
state_dict = torch.load(output_path, map_location="cpu")
parser.model.load_state_dict(state_dict)
```

### 10.6. Toàn bộ luồng train

```text
load_and_preprocess_data()
    -> embeddings + training instances
    -> ParserModel(embeddings)
    -> Adam + CrossEntropyLoss
    -> lặp epoch
        -> model.train()
        -> lặp minibatch
            -> zero_grad
            -> forward
            -> loss
            -> backward
            -> optimizer.step
        -> model.eval()
        -> parse dev set
        -> tính UAS
        -> lưu checkpoint tốt nhất
    -> tải checkpoint tốt nhất
    -> parse test set
    -> test UAS
```

---

## 11. Từ logits đến transition trong `ModelWrapper`

`minibatch_parse()` cần object có method `predict(partial_parses)`. Neural network
lại cần tensor feature IDs. `ModelWrapper` là adapter nối hai giao diện.

### 11.1. Trích feature cho từng parse

```python
mb_x = [
    parser.extract_features(
        parse.stack,
        parse.buffer,
        parse.dependencies,
        dataset[sentence_index],
    )
    for parse in partial_parses
]

mb_x = torch.from_numpy(np.array(mb_x)).long()
```

Shape:

```text
số parse trong minibatch = B
mỗi parse có 36 feature
mb_x.shape = (B, 36)
```

### 11.2. Loại transition không hợp lệ

```python
logits = parser.model(mb_x)
legal = np.array(mb_l).astype("float32")
pred = np.argmax(logits + 10000 * legal, axis=1)
```

`legal` có 1 cho transition hợp lệ và 0 cho transition không hợp lệ. Code hiện
tại cộng `10000` cho transition hợp lệ, khiến chúng thắng transition bất hợp lệ.

Cách biểu diễn rõ ý hơn bằng mask PyTorch:

```python
legal_mask = torch.tensor(mb_l, dtype=torch.bool)
masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
pred = masked_logits.argmax(dim=1)
```

Sau đó đổi class ID thành chuỗi:

```python
transitions = [
    "S" if class_id == 2 else
    "LA" if class_id == 0 else
    "RA"
    for class_id in pred
]
```

---

## 12. Luồng gọi giữa các file

```text
run.py
│
├── load_and_preprocess_data()                  utils/parser_utils.py
│   ├── read_conll()
│   ├── Parser(...)
│   ├── Parser.vectorize()
│   └── Parser.create_instances()
│       ├── extract_features()
│       ├── get_oracle()
│       └── legal_labels()
│
├── ParserModel(...)                            parser_model.py
│   ├── embedding_lookup()
│   └── forward()
│
├── train() / train_for_epoch()                 run.py
│   ├── minibatches()
│   ├── CrossEntropyLoss
│   └── Adam
│
└── Parser.parse(dev/test)                      utils/parser_utils.py
    └── minibatch_parse()                       parser_transitions.py
        └── ModelWrapper.predict()
            ├── extract_features()
            ├── ParserModel.forward()
            └── logits -> S/LA/RA
```

---

## 13. Cách chạy project

Mở terminal tại thư mục chứa các file chính:

```powershell
cd a2\a2
```

### 13.1. Test transition

```powershell
python parser_transitions.py part_c
```

Kiểm tra:

- `SHIFT`;
- `LEFT-ARC`;
- `RIGHT-ARC`;
- parse một chuỗi transition hoàn chỉnh.

### 13.2. Test minibatch parser

```powershell
python parser_transitions.py part_d
```

### 13.3. Test model

```powershell
python parser_model.py -e -f
```

- `-e`: test embedding lookup;
- `-f`: test output shape `(4, 3)`.

### 13.4. Debug training

```powershell
python run.py -d
```

Debug mode chỉ lấy một phần dữ liệu:

```python
train_set = train_set[:1000]
dev_set = dev_set[:500]
test_set = test_set[:500]
```

Nên chạy debug trước vì nhanh hơn và phát hiện phần lớn lỗi shape/training.

### 13.5. Full training

```powershell
python run.py
```

Checkpoint được lưu dưới:

```text
results/YYYYMMDD_HHMMSS/model.weights
```

---

## 14. Tạo môi trường

Từ `a2/a2`:

```powershell
conda env create -f local_env.yml
conda activate cs224n_a2
```

Các dependency chính:

```yaml
dependencies:
  - python>=3.8
  - numpy
  - tqdm
  - docopt
  - pytorch>=2.1.2
  - torchvision
```

Vai trò:

- `numpy`: xử lý dữ liệu và embedding matrix;
- `torch`: xây và train neural network;
- `tqdm`: thanh tiến trình;
- `docopt`: dependency đi kèm bài gốc;
- `torchvision`: không phải thành phần quan trọng của parser này.

---

## 15. `easy_NLP/hw1.py`

File này là ví dụ sentiment analysis dựa trên từ điển, không liên quan trực tiếp
đến dependency parser.

Ý tưởng:

```python
positive_words = ["ngon", "tốt", "thích", "tuyệt", "vui"]
negative_words = ["dở", "tệ", "ghét", "chán", "buồn"]

sentence = "Món ăn này rất ngon và tôi rất thích"
tokens = sentence.lower().split()
```

Đếm số từ tích cực và tiêu cực, sau đó so sánh.

### Lỗi trong code hiện tại

Code hiện tại viết:

```python
for word in range(len(s)):
    if word in tu_tich_cuc:
        ...
```

`range(len(s))` sinh số nguyên `0, 1, 2, ...`, nên `word` là integer, trong khi
`tu_tich_cuc` chứa string. Điều kiện luôn sai.

Phiên bản đúng:

```python
tu_tich_cuc = {"ngon", "tốt", "thích", "tuyệt", "vui"}
tu_tieu_cuc = {"dở", "tệ", "ghét", "chán", "buồn"}

cau_test = "Món ăn này rất ngon và tôi rất thích"
tokens = cau_test.lower().split()

diem_pos = 0
diem_neg = 0

for word in tokens:
    if word in tu_tich_cuc:
        diem_pos += 1
    elif word in tu_tieu_cuc:
        diem_neg += 1

if diem_pos > diem_neg:
    print("tích cực")
elif diem_neg > diem_pos:
    print("tiêu cực")
else:
    print("trung tính")
```

Dùng `set` thay `list` cho từ điển vì phép kiểm tra `word in set` trung bình có
độ phức tạp `O(1)`.

Hạn chế của phương pháp:

- không hiểu phủ định như “không ngon”;
- không xử lý mức độ như “rất tốt”;
- không hiểu ngữ cảnh;
- phụ thuộc hoàn toàn vào từ điển thủ công.

---

## 16. Các lỗi và điểm cần chú ý

### 16.1. Chạy sai working directory

`Config.data_path = "./data"` là đường dẫn tương đối. Hãy chạy từ `a2/a2`, nếu
không chương trình có thể không tìm thấy dữ liệu.

### 16.2. Nhầm class index

Trong cấu hình unlabeled:

```text
0 = LA
1 = RA
2 = S
```

Đổi thứ tự label mà không đổi `ModelWrapper` sẽ tạo kết quả sai dù loss vẫn giảm.

### 16.3. Đưa one-hot label trực tiếp vào CrossEntropyLoss

Code đã đổi đúng:

```python
train_y = train_y.argmax(axis=1)
train_y = torch.from_numpy(train_y).long()
```

### 16.4. Gọi Softmax trước loss

Không làm:

```python
probabilities = torch.softmax(model(x), dim=1)
loss = nn.CrossEntropyLoss()(probabilities, labels)
```

Nên làm:

```python
logits = model(x)
loss = nn.CrossEntropyLoss()(logits, labels)
```

### 16.5. Quên `model.eval()`

Nếu không gọi `.eval()`, Dropout vẫn ngẫu nhiên xóa hidden units khi đánh giá,
làm UAS dao động không cần thiết.

### 16.6. Dùng `.numpy()` với tensor có gradient hoặc ở GPU

Code hiện tại dùng:

```python
pred = pred.detach().numpy()
```

Cách an toàn cho cả GPU:

```python
pred = pred.detach().cpu().numpy()
```

### 16.7. Không dùng `torch.no_grad()` khi đánh giá

`.eval()` không tự tắt autograd. Cách tiết kiệm bộ nhớ hơn:

```python
model.eval()
with torch.no_grad():
    logits = model(features)
```

### 16.8. So sánh version bằng chuỗi

Code:

```python
torch.__version__.split(".") >= ["1", "0", "0"]
```

so sánh string có thể sai với một số version. Nếu thật sự cần kiểm tra version,
nên dùng `packaging.version.Version`.

### 16.9. Không đặt random seed

Embedding ngẫu nhiên, Xavier, dropout và shuffle làm kết quả thay đổi giữa các
lần chạy. Khi debug có thể thêm:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

---

## 17. Một ví dụ PyTorch tối giản mô phỏng model

Đoạn code độc lập sau minh họa đúng luồng shape của `ParserModel` mà không cần
đọc dữ liệu CoNLL:

```python
import torch
import torch.nn as nn


class SmallParserModel(nn.Module):
    def __init__(self, vocab_size, embed_size=50,
                 n_features=36, hidden_size=200, n_classes=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.hidden = nn.Linear(n_features * embed_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(hidden_size, n_classes)

    def forward(self, feature_ids):
        # feature_ids: (B, 36)
        embedded = self.embedding(feature_ids)  # (B, 36, 50)
        flattened = embedded.flatten(1)         # (B, 1800)
        hidden = torch.relu(self.hidden(flattened))  # (B, 200)
        hidden = self.dropout(hidden)                 # (B, 200)
        logits = self.output(hidden)                  # (B, 3)
        return logits


batch_size = 8
vocab_size = 5000

x = torch.randint(
    low=0,
    high=vocab_size,
    size=(batch_size, 36),
    dtype=torch.long,
)

# 0=LA, 1=RA, 2=S
y = torch.randint(0, 3, size=(batch_size,), dtype=torch.long)

model = SmallParserModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss()

model.train()
for step in range(100):
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean()
        print(
            f"step={step:03d} "
            f"loss={loss.item():.4f} "
            f"accuracy={accuracy.item():.2%}"
        )

model.eval()
with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    transitions = logits.argmax(dim=1)
```

Ví dụ này chỉ học phân loại transition. Parser hoàn chỉnh còn cần trạng thái
stack/buffer, feature extraction, legal transition mask và vòng lặp parse.

---

## 18. Checklist để hiểu và tự viết lại bài A2

1. Tự mô phỏng `S`, `LA`, `RA` trên một câu ba từ.
2. Hiểu khi nào parser kết thúc.
3. Viết lại `PartialParse.parse_step()` không nhìn đáp án.
4. Giải thích vì sao `minibatch_parse()` cần giữ cả danh sách gốc và danh sách
   chưa hoàn tất.
5. Liệt kê đủ 18 word features và 18 POS features.
6. Theo dõi shape `(B, 36) -> (B, 36, 50) -> (B, 1800) -> (B, 200) -> (B, 3)`.
7. Hiểu `nn.Parameter` và Xavier initialization.
8. Giải thích vì sao `CrossEntropyLoss` không cần Softmax trước.
9. Nhớ thứ tự `zero_grad -> forward -> loss -> backward -> step`.
10. Phân biệt `model.train()` và `model.eval()`.
11. Hiểu oracle tạo nhãn train như thế nào.
12. Hiểu `ModelWrapper` biến partial parse thành transition ra sao.
13. Biết UAS đo head đúng, không đo dependency label.
14. Chạy được test transition, test model và debug training trước full training.

Nếu nắm được bốn khối `state transition`, `feature extraction`, `neural model`
và `training/evaluation`, bạn đã hiểu toàn bộ kiến trúc của bài A2.
