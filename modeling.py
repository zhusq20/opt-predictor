import torch
from transformers import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')

class CustomOPTModel(torch.nn.Module):
    def __init__(self, model_name):
        super(CustomOPTModel, self).__init__()
        # 加载预配置和预训练的OPT模型
        self.opt_model = OPTForCausalLM.from_pretrained(model_name)

        # 添加一个线性层，这个线性层将第一个token的输出转换为1维
        # 请注意，opt-125m模型的隐层大小是768
        # self.layer = torch.nn.Linear(768, 1)
        self.layer1 = torch.nn.Linear(768, 768 * 4)
        self.gelu = torch.nn.GELU()
        self.sigmoid = torch.nn.Sigmoid()
        self.layer2 = torch.nn.Linear(768 * 4, 1)
        self.tanh = torch.nn.Tanh()
        self.layer = torch.nn.Linear(768, 1)
        # self.layernorm = torch.nn.LayerNorm(768)
        # self.layernorm_input= torch.nn.LayerNorm(768)

    def forward(self, input_ids, attention_mask):
        # 通过OPT模型获取输出
        outputs = self.opt_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)


        # print(outputs.hidden_states[-1].shape)
        # 获取第一个token的输出
        # lst_token_output = outputs.hidden_states[-1][:, -1, :]
        # print(lst_token_output.shape)

        # 找到每个序列的最后一个token的索引
        # attention_mask为1的地方表示有效token，为0表示padding
        # 使用cumsum和max找到每个序列中最后一个1的位置
        seq_lengths = attention_mask.cumsum(dim=1).argmax(dim=1)

        # 从hidden_states中提取每个序列最后一个token的hidden state
        # print(len(outputs.hidden_states))
        last_token_hidden_states = outputs.hidden_states[-2][torch.arange(outputs.hidden_states[-1].size(0)), seq_lengths]
        # print(seq_lengths)

        # 通过预测头进行预测
        # input_norm =self.layernorm_input(last_token_hidden_states)
        # prediction = self.tanh(self.layer2(self.gelu(self.layer1(last_token_hidden_states)))).reshape(1,-1)
        prediction = self.tanh(self.layer(last_token_hidden_states)).reshape(1,-1)
        # prediction = self.tanh(self.layer2(self.sigmoid(self.layer1(last_token_hidden_states)))).reshape(1,-1)

        return prediction

if __name__ == "__main__":
    # 模型名字
    # model_name = 'facebook/opt-125m'

    # 创建模型实例
    model = CustomOPTModel(model_name)

    # 假设你有一些输入数据
    texts = ["Hello, how are you?", "What is your name?", "Transformers are amazing!"]

    encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    # print(encoded_inputs)

    # 获取input_ids和attention_mask
    input_ids = encoded_inputs['input_ids']
    attention_mask = encoded_inputs['attention_mask']
    # print(attention_mask)

    # 进行预测
    prediction = model(input_ids, attention_mask)
    # print("Prediction:", prediction)
