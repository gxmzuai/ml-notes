import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 数据预处理，这个过程需要data.txt文件的路径
def preprocess(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(set(text))
    char2idx = {ch: idx for idx, ch in enumerate(chars)}
    idx2char = {idx: ch for ch, idx in char2idx.items()}
    encoded = np.array([char2idx[ch] for ch in text])
    # 数据集划分
    train_data, test_data = train_test_split(encoded, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
    return train_data, val_data, test_data, char2idx, idx2char


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(
            input_size, hidden_dim, n_layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden


# 初始化模型、损失函数和优化器
def init_model(vocab_size, hidden_dim, n_layers):
    model = LSTMModel(
        input_size=vocab_size,
        output_size=vocab_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return model, criterion, optimizer


# 训练模型
def train(
    model, train_data, val_data, epochs, batch_size, seq_length, criterion, optimizer
):
    model.train()
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        # 训练部分
        model.train()
        total_train_loss = 0
        for i in range(0, len(train_data) - seq_length, batch_size):
            inputs = torch.zeros(seq_length, len(char2idx)).float()
            for j in range(seq_length):
                inputs[j][train_data[i + j]] = 1.0
            inputs = inputs.unsqueeze(0).to(device)
            targets = torch.LongTensor(train_data[i + 1 : i + 1 + seq_length]).to(
                device
            )
            optimizer.zero_grad()
            outputs, hidden = model(inputs, None)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 验证部分
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for i in range(0, len(val_data) - seq_length, batch_size):
                inputs = torch.zeros(seq_length, len(char2idx)).float()
                for j in range(seq_length):
                    inputs[j][val_data[i + j]] = 1.0
                inputs = inputs.unsqueeze(0).to(device)
                targets = torch.LongTensor(val_data[i + 1 : i + 1 + seq_length]).to(
                    device
                )
                outputs, hidden = model(inputs, None)
                val_loss = criterion(outputs, targets.view(-1))
                total_val_loss += val_loss.item()
        print(
            f"Epoch {epoch+1}/{epochs} Training Loss: {total_train_loss / len(train_data):.4f} Validation Loss: {total_val_loss / len(val_data):.4f}"
        )


# 文本生成函数
def generate(model, start_string, char2idx, idx2char, generate_length):
    model.eval()
    # 初始化隐藏状态
    hidden = (
        torch.zeros(model.n_layers, 1, model.hidden_dim).to(device),
        torch.zeros(model.n_layers, 1, model.hidden_dim).to(device),
    )

    # 准备第一个字符的输入张量
    input_idx = char2idx[start_string[-1]]  # 取最后一个字符进行预测
    input_tensor = torch.zeros(1, 1, len(char2idx)).to(device)  # [1, 1, vocab_size]
    input_tensor[0, 0, input_idx] = 1

    output_str = start_string

    for i in range(generate_length):
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)
            output_dist = nn.functional.softmax(output[0], dim=0).data
            top_i = torch.multinomial(output_dist, 1)[0]
            char = idx2char[top_i.item()]
            output_str += char

            input_tensor.fill_(0)  # 清空输入
            input_tensor[0, 0, top_i] = 1  # 更新为最新生成的字符的独热编码

    return output_str


# 实例化并训练模型
train_data, val_data, test_data, char2idx, idx2char = preprocess("data.txt")
model, criterion, optimizer = init_model(
    vocab_size=len(char2idx), hidden_dim=256, n_layers=2
)
train(model, train_data, val_data, 200, 64, 100, criterion, optimizer)

# 保存模型
torch.save(model.state_dict(), "lstm_model.pth")

# 加载模型
model.load_state_dict(torch.load("lstm_model.pth"))

# 监听终端输入
start_string = input("请输入起始歌曲名: ")

# 生成文本
print(
    generate(
        model=model,
        start_string=start_string,
        char2idx=char2idx,
        idx2char=idx2char,
        generate_length=600,
    )
)
