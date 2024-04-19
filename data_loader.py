import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from modeling import CustomOPTModel
from torch.utils.data import DataLoader
from allrank.models.losses.neuralNDCG import neuralNDCG
from allrank.models.losses.listMLE import listMLE
import os
from scipy.stats import kendalltau
# Disable parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Your code that uses tokenizers here

class RankingDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __len2label__(self, length):
            # bucket_size = 16
            # bucket_size_1 = 64
            # if length <=128:
            #     label = 6 + 128 // bucket_size -  length // bucket_size
            # elif length <=512:
            #     label =  512 // bucket_size_1 -  length // bucket_size_1
            # else:
            #     label = 0
            # return label
            label = 512 // 32 -  min(512, length) // 32
            return label

    def __getitem__(self, idx):
        item = self.data[idx]
        # encoded_inputs = self.tokenizer(item['prompt'], max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        # input_ids = encoded_inputs['input_ids'].squeeze(0)
        # attention_mask = encoded_inputs['attention_mask'].squeeze(0)
        prompt = item['prompt']
        label = self.__len2label__(len(self.tokenizer(item['generated'])['input_ids']))
        return prompt, label


if __name__ == "__main__":
    model = CustomOPTModel("/mnt/zsq/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6").to("cuda")
    # opt_tokenizer = AutoTokenizer.from_pretrained('/mnt/zsq/models--facebook--opt-125m/snapshots/27dcfa74d334bc871f3234de431e71c6eeba5dd6')
    tokenizer = AutoTokenizer.from_pretrained("/mnt/zsq/model_13B/models--lmsys--vicuna-13b-v1.5")
    opt_tokenizer = tokenizer
    # Create the DataLoader
    import torch
    batch_size = 256
    micro_batch = 2

    all_data = []
    train_set = []
    test_set = []
    with open("/mnt/zsq/vllm-scheduler-dev/vllm-0416/benchmarks/alpaca-vicuna-13b-v1.5-t1.0-s0-l4096-c20000:30000-rFalse.jsonl") as f:
        for jsonObj in f:
            info = json.loads(jsonObj)
            all_data.append(info)
    train_set = all_data[:int(0.8 * len(all_data))]
    test_set = all_data[int(0.8 * len(all_data)):]
    train_dataset = RankingDataset(train_set, tokenizer)
    test_dataset = RankingDataset(test_set, tokenizer)
     
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    # neural_ndcg_loss = neuralNDCG()
    # Training loop
    for epoch in range(50):
        model.train()
        total_loss = 0
        for prompt, labels in train_dataloader:
            # print(prompt) 
            prompt = list(prompt)
            # 将元组转换成2*8的二维列表
            # prompt = [list(prompt[i:i+2]) for i in range(0, len(prompt), 2)]

            encoded_inputs = opt_tokenizer(prompt, max_length=1024, padding=True, truncation=True, return_tensors="pt")
            # print(encoded_inputs)
            input_ids = encoded_inputs['input_ids'].to("cuda:0")
            attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)

            labels = torch.tensor(labels, dtype=torch.float32)
            labels = labels.requires_grad_(True)  # Note: Changed to requires_grad_() for in-place operation
            labels = labels.reshape(micro_batch, -1)
            labels = labels.to("cuda")

            loss = listMLE(outputs.view(micro_batch, -1), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

        true_labels = []
        predictions = []
        
        model.eval()
        with torch.no_grad():
            for prompt, labels in test_dataloader:
                prompt = list(prompt)
                # 将元组转换成2*8的二维列表
                # prompt = [list(prompt[i:i+2]) for i in range(0, len(prompt), 2)]

                encoded_inputs = opt_tokenizer(prompt, max_length=1024, padding=True, truncation=True, return_tensors="pt")
                # print(encoded_inputs)
                input_ids = encoded_inputs['input_ids'].to("cuda:0")
                attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
                outputs = model(input_ids, attention_mask)
            
                # 假设输出是单值预测
                predicted_scores = outputs.squeeze().tolist()
                true_labels.extend(labels.tolist())
                predictions.extend(predicted_scores)
            # 计算 Kendall's Tau
            tau, score = kendalltau(true_labels, predictions)
            # return tau
            # print("tau, score",tau, score)
            print(f"Kendall's Tau: {tau}, p-value: {score}")



