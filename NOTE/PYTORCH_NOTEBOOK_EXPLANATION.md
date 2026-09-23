# Giải thích PyTorch Tutorial và cách tự xây dựng mạng nơ-ron/RNN

Tài liệu này giải thích các phần quan trọng trong notebook
`SP_24_CS224N_PyTorch_Tutorial.ipynb`, tập trung vào:

- cú pháp PyTorch cơ bản;
- cách dữ liệu đi qua một mạng nơ-ron;
- cách tự định nghĩa mạng bằng class kế thừa `nn.Module`;
- mô hình `WordWindowClassifier` ở cuối notebook;
- cách tự cài đặt một RNN và cách dùng `nn.RNN` của PyTorch.

> **Lưu ý:** Mạng ở cuối notebook là **Word Window Classifier**, không phải RNN.
> Nó nhìn một cửa sổ từ cố định quanh mỗi từ rồi phân loại. Phần RNN được bổ sung
> ở cuối tài liệu để cho thấy cách một mạng tuần tự thực sự lưu trạng thái từ các
> bước thời gian trước.

---

## 1. Quy ước kích thước tensor

Khi đọc code mạng nơ-ron, nên ghi chú kích thước tensor ở từng dòng:

| Ký hiệu | Ý nghĩa |
|---|---|
| `B` | batch size — số câu/mẫu trong một batch |
| `L` | sequence length — số token trong một câu |
| `V` | vocabulary size — số từ trong từ điển |
| `D` | embedding dimension — số chiều vector biểu diễn một từ |
| `H` | hidden dimension — số chiều trạng thái ẩn |
| `C` | số lớp cần dự đoán |
| `S` | window size; cửa sổ đầy đủ có `2*S + 1` từ |

Ví dụ, tensor có shape `(32, 20, 100)` có thể được hiểu là 32 câu, mỗi câu 20
token, mỗi token được biểu diễn bằng vector 100 chiều.

---

## 2. Các thao tác PyTorch cơ bản

### 2.1. Import thư viện

```python
import torch
import torch.nn as nn
import torch.optim as optim
```

- `torch`: tensor, phép toán số học và autograd.
- `torch.nn`: layer, activation, loss và class nền `nn.Module`.
- `torch.optim`: các thuật toán cập nhật tham số như SGD, Adam.

### 2.2. Tạo tensor

```python
x = torch.tensor([[1, 2], [3, 4]])
x_float = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)

zeros = torch.zeros(2, 3)       # shape (2, 3), toàn số 0
ones = torch.ones(2, 3)         # shape (2, 3), toàn số 1
random = torch.randn(2, 3)      # phân phối chuẩn N(0, 1)
indices = torch.arange(0, 10)   # 0, 1, ..., 9
```

Các kiểu dữ liệu thường dùng:

- `torch.float32`: trọng số, embedding, đầu vào số thực;
- `torch.long` hay `torch.int64`: chỉ số token đưa vào `nn.Embedding`;
- `torch.bool`: mask đúng/sai.

```python
token_ids = torch.tensor([2, 5, 7], dtype=torch.long)
```

### 2.3. Shape và đổi shape

```python
x = torch.arange(12)
print(x.shape)       # torch.Size([12])

x = x.reshape(3, 4)
print(x.shape)       # torch.Size([3, 4])

x = x.view(2, 6)
print(x.shape)       # torch.Size([2, 6])
```

`reshape` và `view` đều đổi hình dạng mà không đổi số phần tử. `reshape` thường
an toàn hơn vì có thể xử lý tensor không liên tục trong bộ nhớ. `-1` yêu cầu
PyTorch tự suy ra chiều còn lại:

```python
x = torch.randn(8, 5, 10)  # (B=8, L=5, D=10)
x = x.reshape(8, 5, -1)    # chiều cuối vẫn là 10
```

Thêm hoặc bỏ một chiều:

```python
x = torch.tensor([1, 2, 3])  # (3,)
x = x.unsqueeze(0)           # (1, 3)
x = x.unsqueeze(-1)          # (1, 3, 1)
x = x.squeeze(-1)            # (1, 3)
```

Đổi thứ tự chiều:

```python
x = torch.randn(32, 20, 100)  # (B, L, D)
y = x.transpose(0, 1)         # (L, B, D)
y = x.permute(1, 0, 2)        # (L, B, D)
```

### 2.4. Indexing và slicing

```python
x = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
    [[9, 10], [11, 12]],
])  # shape (3, 2, 2)

x[0]          # phần tử đầu theo chiều 0, shape (2, 2)
x[:, 0]       # hàng đầu của mọi khối, shape (3, 2)
x[:, 0, 0]    # phần tử góc trái của mỗi khối, shape (3,)
x[0, 0, 0].item()  # chuyển tensor vô hướng thành số Python
```

`:` nghĩa là lấy toàn bộ phần tử trên chiều tương ứng.

### 2.5. Phép toán

```python
x + 2             # cộng từng phần tử
x * 2             # nhân từng phần tử
x.sum()           # tổng tất cả phần tử
x.sum(dim=0)      # cộng theo chiều 0
x.mean(dim=1)     # trung bình theo chiều 1
```

Phân biệt nhân từng phần tử và nhân ma trận:

```python
a * b             # element-wise; a và b phải broadcast được
a @ b             # nhân ma trận
torch.matmul(a, b)
```

Nếu `a` có shape `(3, 2)` và `b` có shape `(2, 4)`, `a @ b` có shape `(3, 4)`.

### 2.6. Nối và xếp tensor

```python
a = torch.randn(2, 3)
b = torch.randn(2, 3)

torch.cat([a, b], dim=0).shape    # (4, 3), nối trên chiều có sẵn
torch.stack([a, b], dim=0).shape  # (2, 2, 3), tạo chiều mới
```

Trong RNN tự cài đặt, `torch.cat([x_t, h_prev], dim=1)` thường được dùng để ghép
đầu vào hiện tại với trạng thái ẩn trước đó.

### 2.7. NumPy và device CPU/GPU

```python
import numpy as np

array = np.array([[1, 2, 3]])
x = torch.from_numpy(array)
array_again = x.numpy()
```

Chọn thiết bị:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
model = model.to(device)
```

Model và tensor đầu vào phải ở cùng device. Khi chuyển tensor đang giữ gradient
sang NumPy, dùng:

```python
array = x.detach().cpu().numpy()
```

---

## 3. Autograd và lan truyền ngược

PyTorch xây dựng đồ thị tính toán động để tự tính đạo hàm.

```python
x = torch.tensor([2.0], requires_grad=True)
y = 3 * x * x
y.backward()
print(x.grad)  # 12, vì dy/dx = 6x và x = 2
```

Ý nghĩa các thành phần:

- `requires_grad=True`: yêu cầu PyTorch theo dõi phép toán trên tensor;
- `loss.backward()`: tính gradient của loss theo tất cả tham số liên quan;
- `parameter.grad`: nơi lưu gradient;
- `optimizer.step()`: cập nhật tham số dựa trên gradient.

Gradient **được cộng dồn** qua nhiều lần gọi `backward()`:

```python
y = 3 * x * x
y.backward()
print(x.grad)  # 12

z = 3 * x * x
z.backward()
print(x.grad)  # 24, không phải 12
```

Vì vậy mỗi vòng huấn luyện phải xóa gradient cũ:

```python
optimizer.zero_grad()
```

Khi suy luận, không cần xây đồ thị gradient:

```python
model.eval()
with torch.no_grad():
    predictions = model(x)
```

`model.eval()` chuyển các layer có hành vi train/eval khác nhau như Dropout và
BatchNorm sang chế độ suy luận. `torch.no_grad()` tắt việc lưu đồ thị gradient.

---

## 4. Một neuron và mạng nơ-ron hoạt động như thế nào?

Một neuron tuyến tính tính:

\[
z = xW^T + b
\]

sau đó có thể đi qua hàm kích hoạt:

\[
h = \tanh(z) \quad \text{hoặc} \quad h = \operatorname{ReLU}(z)
\]

Trong PyTorch:

```python
layer = nn.Linear(in_features=4, out_features=3)
x = torch.randn(8, 4)  # 8 mẫu, mỗi mẫu 4 đặc trưng
z = layer(x)            # shape (8, 3)
h = torch.relu(z)       # shape không đổi
```

`nn.Linear(4, 3)` tự tạo:

- `weight` có shape `(3, 4)`;
- `bias` có shape `(3,)`.

Có thể xem chúng bằng:

```python
print(layer.weight.shape)
print(layer.bias.shape)
for name, parameter in layer.named_parameters():
    print(name, parameter.shape)
```

---

## 5. Tự cài đặt một lớp Linear

Phần này cho thấy `nn.Linear` thực chất chỉ là tham số `W`, `b` và phép nhân ma
trận. Ta vẫn dùng autograd của PyTorch, nhưng tự viết phép tính forward.

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: (..., in_features)
        # weight.T: (in_features, out_features)
        # output: (..., out_features)
        return x @ self.weight.T + self.bias
```

Các điểm cú pháp quan trọng:

1. `class MyLinear(nn.Module)` để PyTorch biết đây là một module.
2. `super().__init__()` phải chạy trước khi gắn layer/parameter vào `self`.
3. `nn.Parameter(tensor)` đăng ký tensor làm tham số có thể học.
4. `forward(self, x)` mô tả dữ liệu đi qua module như thế nào.
5. Gọi `layer(x)`, không nên gọi trực tiếp `layer.forward(x)`. Cú pháp `layer(x)`
   cho phép PyTorch chạy hook và các logic nội bộ của `nn.Module`.

Kiểm tra:

```python
layer = MyLinear(4, 3)
x = torch.randn(8, 4)
y = layer(x)
print(y.shape)  # torch.Size([8, 3])
```

### Tự cập nhật tham số mà không dùng optimizer

Đây là cách SGD hoạt động ở mức cơ bản:

```python
learning_rate = 0.01

loss.backward()

with torch.no_grad():
    for parameter in model.parameters():
        parameter -= learning_rate * parameter.grad
        parameter.grad.zero_()
```

Không được cập nhật trực tiếp tham số trong lúc autograd đang theo dõi, vì thao
tác cập nhật không phải là một phần của mô hình cần lấy đạo hàm. Do đó dùng
`with torch.no_grad()`.

Trong dự án thực tế, nên dùng optimizer:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 6. Xây mạng MLP bằng class

### 6.1. Dùng `nn.Sequential`

```python
class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)
```

`nn.Sequential` thích hợp khi dữ liệu chỉ đi lần lượt từ layer này sang layer
khác.

### 6.2. Khai báo từng layer riêng

```python
class MultilayerPerceptron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
```

Cách thứ hai dài hơn nhưng dễ kiểm tra shape, thêm skip connection, tạo nhánh
hoặc áp dụng logic tùy biến.

Ví dụ sử dụng:

```python
model = MultilayerPerceptron(
    input_size=5,
    hidden_size=10,
    output_size=2,
)

x = torch.randn(32, 5)
logits = model(x)
print(logits.shape)  # (32, 2)
```

Nếu dùng `nn.CrossEntropyLoss`, output cuối là **logits**, không thêm Softmax:

```python
labels = torch.randint(0, 2, size=(32,))
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)
```

`CrossEntropyLoss` đã kết hợp `LogSoftmax` và negative log-likelihood một cách
ổn định số học.

---

## 7. Quy trình huấn luyện chuẩn

```python
model = MultilayerPerceptron(5, 10, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()       # 1. xóa gradient cũ
    logits = model(x)           # 2. forward
    loss = criterion(logits, labels)  # 3. tính loss
    loss.backward()             # 4. tính gradient
    optimizer.step()            # 5. cập nhật tham số

    if epoch % 10 == 0:
        print(epoch, loss.item())
```

Thứ tự cần nhớ:

```text
zero_grad -> forward -> loss -> backward -> step
```

### `model.train()` và `model.eval()`

```python
model.train()
# huấn luyện

model.eval()
with torch.no_grad():
    # validation/test
```

Hai lệnh này đặc biệt quan trọng nếu model có `nn.Dropout` hoặc BatchNorm.

### Lưu và tải model

```python
torch.save(model.state_dict(), "model.pt")

model = MultilayerPerceptron(5, 10, 2)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()
```

---

## 8. Tiền xử lý NLP trong notebook

### 8.1. Tokenization đơn giản

```python
def preprocess_sentence(sentence):
    return sentence.lower().split()
```

Ví dụ `"She comes from Paris"` trở thành:

```python
["she", "comes", "from", "paris"]
```

Đây chỉ là tokenizer minh họa. Dữ liệu thật có dấu câu, từ ghép hoặc nhiều ngôn
ngữ thường cần tokenizer chuyên dụng.

### 8.2. Tạo vocabulary

```python
vocabulary = set(word for sentence in train_sentences for word in sentence)
vocabulary.add("<unk>")
vocabulary.add("<pad>")

ix_to_word = sorted(vocabulary)
word_to_ix = {word: index for index, word in enumerate(ix_to_word)}
```

- `<unk>` thay cho từ chưa từng xuất hiện trong vocabulary.
- `<pad>` lấp chỗ trống để các câu trong batch có cùng chiều dài.

Chuyển token thành chỉ số:

```python
def convert_tokens_to_indices(sentence, word_to_ix):
    unk_ix = word_to_ix["<unk>"]
    return [word_to_ix.get(token, unk_ix) for token in sentence]
```

Trong notebook, phiên bản rút gọn dùng nhầm tên `word_to_ind`; tên đúng phải là
`word_to_ix`.

### 8.3. Embedding

```python
embedding = nn.Embedding(
    num_embeddings=len(word_to_ix),
    embedding_dim=50,
    padding_idx=word_to_ix["<pad>"],
)

token_ids = torch.tensor([[1, 4, 8], [2, 5, 0]], dtype=torch.long)
vectors = embedding(token_ids)
print(vectors.shape)  # (2, 3, 50)
```

`nn.Embedding` là một bảng trọng số shape `(V, D)`. Mỗi token ID chọn một hàng
trong bảng. Trọng số embedding cũng được học qua backpropagation.

`padding_idx` giữ vector padding cố định và không cập nhật nó khi huấn luyện.

### 8.4. Padding và DataLoader

```python
from torch.nn.utils.rnn import pad_sequence

sentences = [
    torch.tensor([2, 4, 6], dtype=torch.long),
    torch.tensor([1, 3], dtype=torch.long),
]

padded = pad_sequence(
    sentences,
    batch_first=True,
    padding_value=0,
)
# tensor([[2, 4, 6],
#         [1, 3, 0]])
```

`DataLoader` chia dữ liệu thành batch:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=custom_collate_fn,
)
```

`collate_fn` nhận một danh sách mẫu và biến chúng thành tensor batch. Trong
notebook, nó thực hiện:

1. tách câu và nhãn;
2. thêm `<pad>` ở hai đầu cho word window;
3. đổi token thành index;
4. pad các câu về cùng độ dài;
5. trả về input, label và độ dài thật.

---

## 9. Giải thích `WordWindowClassifier` cuối notebook

### 9.1. Ý tưởng

Với `window_size = 2`, mỗi từ được phân loại bằng 5 từ:

```text
[t-2, t-1, t, t+1, t+2]
```

Ví dụ khi dự đoán từ `paris` trong `we come to paris`, câu được pad để tạo đủ
ngữ cảnh:

```text
<pad> <pad> we come to paris <pad> <pad>
```

`unfold` tạo mọi cửa sổ mà không cần vòng lặp Python:

```python
windows = inputs.unfold(
    dimension=1,
    size=2 * window_size + 1,
    step=1,
)
```

Nếu `inputs` có shape `(B, L_padded)`, `windows` có shape:

```text
(B, L_original, 2 * window_size + 1)
```

### 9.2. Phiên bản class đã chỉnh cú pháp

```python
class WordWindowClassifier(nn.Module):
    def __init__(self, hyperparameters, vocab_size, pad_ix):
        super().__init__()

        self.window_size = hyperparameters["window_size"]
        self.embed_dim = hyperparameters["embed_dim"]
        self.hidden_dim = hyperparameters["hidden_dim"]

        self.embeds = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embed_dim,
            padding_idx=pad_ix,
        )

        if hyperparameters["freeze_embeddings"]:
            self.embeds.weight.requires_grad = False

        full_window_size = 2 * self.window_size + 1

        self.hidden_layer = nn.Sequential(
            nn.Linear(full_window_size * self.embed_dim, self.hidden_dim),
            nn.Tanh(),
        )

        self.output_layer = nn.Linear(self.hidden_dim, 1)

    def forward(self, inputs):
        # inputs: (B, L_padded)
        batch_size = inputs.size(0)

        # (B, L_original, full_window_size)
        token_windows = inputs.unfold(
            dimension=1,
            size=2 * self.window_size + 1,
            step=1,
        )

        # (B, L_original, full_window_size, D)
        embedded_windows = self.embeds(token_windows)

        # (B, L_original, full_window_size * D)
        embedded_windows = embedded_windows.reshape(
            batch_size,
            token_windows.size(1),
            -1,
        )

        # (B, L_original, H)
        hidden = self.hidden_layer(embedded_windows)

        # (B, L_original), trả về logits
        logits = self.output_layer(hidden).squeeze(-1)
        return logits
```

Các điểm đã sửa so với notebook:

- dùng `self.window_size` thay vì biến toàn cục `window_size`;
- dùng `self.embeds.weight` thay vì thuộc tính không tồn tại `self.embed_layer`;
- model trả về logits, không gọi Sigmoid bên trong;
- dùng `BCEWithLogitsLoss` để ổn định số học hơn `Sigmoid + BCELoss`.

### 9.3. Mask padding khi tính loss

Padding của label không phải dữ liệu thật, vì vậy không nên đưa vào loss. Có thể
tạo mask từ độ dài câu:

```python
def masked_bce_loss(logits, labels, lengths):
    # logits, labels: (B, L)
    max_length = labels.size(1)
    positions = torch.arange(max_length, device=labels.device)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)  # (B, L)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    losses = criterion(logits, labels.float())

    return (losses * mask).sum() / mask.sum()
```

Chỉ chia toàn bộ `BCELoss` cho tổng độ dài như notebook chưa loại ảnh hưởng của
vị trí padding. Dùng mask là cách chính xác hơn.

### 9.4. Training loop

```python
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for inputs, labels, lengths in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = masked_bce_loss(logits, labels, lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
```

Khi dự đoán, đổi logits thành xác suất và nhãn:

```python
model.eval()
with torch.no_grad():
    logits = model(inputs)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).long()
```

### 9.5. Vì sao đây chưa phải RNN?

Mỗi cửa sổ được đưa qua cùng một MLP một cách độc lập. Model không truyền một
trạng thái ẩn từ từ đầu câu đến từ cuối câu. Ngữ cảnh bị giới hạn ở đúng
`2 * window_size + 1` token. RNN thì cập nhật trạng thái ẩn tuần tự:

\[
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
\]

Do `h_t` phụ thuộc `h_{t-1}`, thông tin từ các token trước có thể tiếp tục đi
qua chuỗi.

---

## 10. Tự tạo một RNN thủ công bằng class

Đây là phần quan trọng nhất nếu muốn hiểu bản chất RNN.

### 10.1. Công thức một RNN cell

Tại thời điểm `t`:

\[
h_t = \tanh(x_t W_x^T + h_{t-1} W_h^T + b_h)
\]

Nếu cần dự đoán tại mỗi token:

\[
y_t = h_t W_y^T + b_y
\]

Trong đó:

- `x_t`: input tại bước `t`, shape `(B, D)`;
- `h_{t-1}`: hidden state cũ, shape `(B, H)`;
- `h_t`: hidden state mới, shape `(B, H)`;
- `y_t`: logits tại bước `t`, shape `(B, C)`.

### 10.2. Tự viết `RNNCell`

Phiên bản dễ hiểu dùng hai lớp Linear:

```python
class MyRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # x_t -> phần đóng góp vào hidden mới
        self.input_to_hidden = nn.Linear(
            input_size,
            hidden_size,
            bias=True,
        )

        # h_(t-1) -> phần đóng góp vào hidden mới
        # Không cần bias thứ hai vì bias ở layer trên đã đủ.
        self.hidden_to_hidden = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

    def forward(self, x_t, h_prev):
        # x_t:    (B, D)
        # h_prev: (B, H)
        h_t = torch.tanh(
            self.input_to_hidden(x_t)
            + self.hidden_to_hidden(h_prev)
        )
        return h_t
```

Cách khác là nối hai tensor rồi dùng một lớp Linear:

```python
class MyRNNCellConcat(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.combined_to_hidden = nn.Linear(
            input_size + hidden_size,
            hidden_size,
        )

    def forward(self, x_t, h_prev):
        combined = torch.cat([x_t, h_prev], dim=1)
        return torch.tanh(self.combined_to_hidden(combined))
```

Hai cách biểu diễn cùng một ý tưởng. Cách tách hai Linear bám sát công thức
`W_x x_t + W_h h_(t-1)` hơn.

### 10.3. Lặp qua toàn bộ chuỗi

```python
class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = MyRNNCell(input_size, hidden_size)

    def forward(self, x, h_0=None):
        # x: (B, L, D) vì dùng batch_first
        batch_size, sequence_length, _ = x.shape

        if h_0 is None:
            h_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            h_t = h_0

        hidden_states = []

        for t in range(sequence_length):
            x_t = x[:, t, :]       # (B, D)
            h_t = self.cell(x_t, h_t)  # (B, H)
            hidden_states.append(h_t)

        # Danh sách L tensor (B, H) -> một tensor (B, L, H)
        outputs = torch.stack(hidden_states, dim=1)

        # outputs: hidden state của mọi bước
        # h_t: hidden state cuối
        return outputs, h_t
```

Các dòng đáng nhớ:

```python
x_t = x[:, t, :]
```

Lấy input của mọi mẫu trong batch tại thời điểm `t`.

```python
hidden_states.append(h_t)
outputs = torch.stack(hidden_states, dim=1)
```

Lưu hidden state của từng thời điểm rồi ghép thành `(B, L, H)`. Dùng `stack`,
không dùng `torch.tensor(hidden_states)`, vì cách thứ hai làm mất đồ thị gradient.

### 10.4. Thêm embedding và đầu ra để phân loại từng token

Ví dụ bài toán giống notebook: dự đoán mỗi token có phải địa điểm hay không.

```python
class ManualRNNTokenClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, pad_ix):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_ix,
        )
        self.rnn = MyRNN(embedding_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, token_ids):
        # token_ids: (B, L)
        embedded = self.embedding(token_ids)  # (B, L, D)
        outputs, final_hidden = self.rnn(embedded)  # (B, L, H), (B, H)
        logits = self.classifier(outputs).squeeze(-1)  # (B, L)
        return logits
```

Luồng kích thước:

```text
token IDs       (B, L)
    -> Embedding (B, L, D)
    -> MyRNN     (B, L, H)
    -> Linear    (B, L, 1)
    -> squeeze   (B, L)
```

Khởi tạo và gọi model:

```python
model = ManualRNNTokenClassifier(
    vocab_size=len(word_to_ix),
    embedding_dim=50,
    hidden_size=64,
    pad_ix=word_to_ix["<pad>"],
)

token_ids = torch.randint(0, len(word_to_ix), size=(4, 10))
logits = model(token_ids)
print(logits.shape)  # (4, 10)
```

### 10.5. Huấn luyện RNN tự viết

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    total_loss = 0.0

    for token_ids, labels, lengths in loader:
        token_ids = token_ids.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        logits = model(token_ids)
        loss = masked_bce_loss(logits, labels, lengths)
        loss.backward()

        # RNN có thể bị exploding gradient.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d} | loss = {total_loss / len(loader):.4f}")
```

`clip_grad_norm_` giới hạn chuẩn gradient. Nó thường hữu ích với RNN vì cùng một
phép biến đổi trạng thái được áp dụng lặp lại nhiều lần.

### 10.6. RNN phân loại cả câu

Nếu chỉ cần một nhãn cho cả câu, có thể dùng hidden state cuối:

```python
class ManualRNNSentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 num_classes, pad_ix):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_ix,
        )
        self.rnn = MyRNN(embedding_dim, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)
        _, final_hidden = self.rnn(embedded)
        logits = self.classifier(final_hidden)
        return logits
```

Với batch chứa padding, `final_hidden` ở trên có thể tương ứng với bước padding,
không phải token thật cuối cùng. Cách đơn giản là lấy hidden state theo `lengths`:

```python
outputs, _ = self.rnn(embedded)  # (B, L, H)
batch_indices = torch.arange(outputs.size(0), device=outputs.device)
last_hidden = outputs[batch_indices, lengths - 1]
logits = self.classifier(last_hidden)
```

---

## 11. Dùng `nn.RNN` của PyTorch

Sau khi hiểu bản thủ công, code thực tế thường dùng `nn.RNN`, `nn.GRU` hoặc
`nn.LSTM` vì chúng được tối ưu tốt hơn.

### 11.1. Class hoàn chỉnh dùng `nn.RNN`

```python
class PyTorchRNNTokenClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, pad_ix):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_ix,
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, token_ids):
        # token_ids: (B, L)
        embedded = self.embedding(token_ids)       # (B, L, D)
        outputs, h_n = self.rnn(embedded)          # (B, L, H), (1, B, H)
        logits = self.classifier(outputs)          # (B, L, 1)
        logits = logits.squeeze(-1)                # (B, L)
        return logits
```

### 11.2. Tham số quan trọng của `nn.RNN`

```python
nn.RNN(
    input_size=D,
    hidden_size=H,
    num_layers=1,
    nonlinearity="tanh",
    batch_first=True,
    bidirectional=False,
    dropout=0.0,
)
```

- `input_size`: số chiều của vector tại mỗi thời điểm;
- `hidden_size`: số chiều hidden state;
- `num_layers`: số tầng RNN xếp chồng;
- `batch_first=True`: input/output có dạng `(B, L, D/H)`;
- `bidirectional=True`: chạy cả trái sang phải và phải sang trái;
- `dropout`: dropout giữa các tầng, chỉ có tác dụng khi `num_layers > 1`.

Kết quả:

```python
outputs, h_n = self.rnn(embedded)
```

- `outputs`: hidden state của tầng cuối tại mọi thời điểm;
- `h_n`: hidden state cuối của từng tầng/hướng.

Với RNN một tầng, một chiều:

```text
outputs: (B, L, H)
h_n:     (1, B, H)
```

### 11.3. Truyền hidden state ban đầu

Nếu không truyền, PyTorch tự dùng tensor 0. Có thể truyền rõ ràng:

```python
h_0 = torch.zeros(
    1,                  # num_layers * num_directions
    batch_size,
    hidden_size,
    device=embedded.device,
)

outputs, h_n = self.rnn(embedded, h_0)
```

### 11.4. RNN hai chiều

```python
self.rnn = nn.RNN(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    batch_first=True,
    bidirectional=True,
)

self.classifier = nn.Linear(hidden_size * 2, 1)
```

Vì ghép output từ hai hướng, chiều cuối của `outputs` là `2 * H`:

```text
outputs: (B, L, 2H)
h_n:     (2, B, H)   # 1 layer x 2 hướng
```

RNN hai chiều phù hợp với gán nhãn token khi toàn bộ câu đã có sẵn, vì dự đoán
một token có thể dùng cả ngữ cảnh bên trái và bên phải.

### 11.5. Xử lý câu có độ dài khác nhau

Mask loss đã ngăn padding ảnh hưởng trực tiếp tới loss. Để RNN không tính cả các
bước padding, có thể dùng packed sequence:

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

embedded = self.embedding(token_ids)

packed = pack_padded_sequence(
    embedded,
    lengths.cpu(),
    batch_first=True,
    enforce_sorted=False,
)

packed_outputs, h_n = self.rnn(packed)

outputs, _ = pad_packed_sequence(
    packed_outputs,
    batch_first=True,
)
```

`lengths.cpu()` là yêu cầu thường gặp của hàm packing. `enforce_sorted=False`
cho phép batch chưa được sắp xếp theo chiều dài giảm dần.

---

## 12. So sánh Word Window, RNN, GRU và LSTM

| Mô hình | Ngữ cảnh | Ưu điểm | Nhược điểm |
|---|---|---|---|
| Word Window MLP | cửa sổ cố định | đơn giản, dễ song song hóa | không nhớ ngữ cảnh xa |
| Vanilla RNN | toàn bộ chuỗi trước đó | dễ hiểu, ít tham số | dễ vanishing/exploding gradient |
| GRU | toàn bộ chuỗi, có gate | gọn hơn LSTM, nhớ tốt hơn RNN | phức tạp hơn RNN |
| LSTM | toàn bộ chuỗi, có cell state | kiểm soát nhớ/quên tốt | nhiều tham số hơn |

Thay `nn.RNN` bằng GRU thường chỉ cần:

```python
self.rnn = nn.GRU(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    batch_first=True,
)
```

LSTM trả về thêm cell state:

```python
self.rnn = nn.LSTM(
    input_size=embedding_dim,
    hidden_size=hidden_size,
    batch_first=True,
)

outputs, (h_n, c_n) = self.rnn(embedded)
```

---

## 13. Lỗi thường gặp

### Sai dtype cho `nn.Embedding`

```python
# Sai: FloatTensor
token_ids = torch.tensor([1, 2, 3], dtype=torch.float32)

# Đúng: LongTensor
token_ids = torch.tensor([1, 2, 3], dtype=torch.long)
```

### Sai shape của Linear

`nn.Linear(D, H)` luôn yêu cầu chiều cuối của input bằng `D`.

```python
layer = nn.Linear(10, 20)
x = torch.randn(32, 5, 10)  # hợp lệ -> (32, 5, 20)
```

### Quên `super().__init__()`

Nếu quên, các submodule và parameter có thể không được đăng ký đúng.

### Gọi `forward` trực tiếp

```python
output = model(x)          # nên dùng
output = model.forward(x)  # tránh dùng trực tiếp
```

### Quên xóa gradient

Gradient bị cộng dồn nếu không gọi `optimizer.zero_grad()`.

### Thêm activation không phù hợp trước loss

- `CrossEntropyLoss`: truyền logits, không Softmax trước.
- `BCEWithLogitsLoss`: truyền logits, không Sigmoid trước.
- Khi dự đoán mới dùng Softmax/Sigmoid để đổi logits thành xác suất.

### Không mask padding

Nếu label padding đi vào loss, model sẽ học cả dữ liệu giả. Hãy dùng mask hoặc
`ignore_index` nếu loss hỗ trợ.

### Tạo hidden state sai device

```python
# Có thể sai khi x ở GPU
h = torch.zeros(batch_size, hidden_size)

# Đúng
h = torch.zeros(batch_size, hidden_size, device=x.device, dtype=x.dtype)
```

### Nhầm `detach()` với `no_grad()`

- `tensor.detach()` tạo tensor dùng chung dữ liệu nhưng tách khỏi đồ thị hiện tại.
- `with torch.no_grad()` tắt theo dõi gradient cho các phép toán trong khối.

### In-place operation làm hỏng autograd

Các thao tác có dấu gạch dưới như `add_`, `zero_`, `relu_` sửa tensor tại chỗ.
Không dùng chúng tùy tiện trên tensor đang cần cho backward.

---

## 14. Mẫu hoàn chỉnh tối giản để tự thực hành RNN

```python
import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,
                 num_classes, pad_ix):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=pad_ix,
        )
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)  # (B, L, D)
        _, h_n = self.rnn(embedded)            # h_n: (1, B, H)
        final_hidden = h_n[-1]                 # (B, H)
        logits = self.output(final_hidden)     # (B, C)
        return logits


# Dữ liệu minh họa: 4 câu, mỗi câu đã pad về 6 token
vocab_size = 100
pad_ix = 0
x = torch.randint(1, vocab_size, size=(4, 6), dtype=torch.long)
y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

model = RNNClassifier(
    vocab_size=vocab_size,
    embedding_dim=16,
    hidden_size=32,
    num_classes=2,
    pad_ix=pad_ix,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    logits = model(x)
    loss = criterion(logits, y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if epoch % 10 == 0:
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean()
        print(
            f"epoch={epoch:03d} "
            f"loss={loss.item():.4f} "
            f"accuracy={accuracy.item():.2%}"
        )

model.eval()
with torch.no_grad():
    logits = model(x)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
```

Để biến ví dụ trên thành bài toán thực tế cần thêm:

1. tokenizer;
2. vocabulary và ánh xạ `word_to_ix`;
3. train/validation/test split;
4. `Dataset` và `DataLoader`;
5. padding/mask hoặc packed sequence;
6. metric phù hợp với bài toán;
7. lưu checkpoint tốt nhất theo validation loss.

---

## 15. Checklist khi tự viết một model PyTorch

1. Xác định shape input và output mong muốn.
2. Kế thừa `nn.Module`.
3. Gọi `super().__init__()`.
4. Khai báo layer hoặc `nn.Parameter` trong `__init__`.
5. Viết luồng dữ liệu trong `forward`.
6. Ghi chú shape sau từng phép biến đổi.
7. Chọn loss phù hợp với logits và kiểu nhãn.
8. Tạo optimizer từ `model.parameters()`.
9. Huấn luyện theo thứ tự `zero_grad -> forward -> loss -> backward -> step`.
10. Mask padding cho dữ liệu chuỗi.
11. Dùng `model.eval()` và `torch.no_grad()` khi đánh giá.
12. Kiểm tra model và dữ liệu ở cùng device.

Nếu hiểu được đường đi của shape và vai trò của từng dòng trong `forward`, bạn
đã nắm phần cốt lõi để tự xây dựng MLP, RNN, GRU hoặc LSTM bằng PyTorch.
