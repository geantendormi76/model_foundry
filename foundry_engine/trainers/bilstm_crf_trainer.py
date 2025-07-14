# foundry_engine/trainers/bilstm_crf_trainer.py (V4 - 解耦导出逻辑)

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from typing import Type, Dict, List, Tuple
from tqdm import tqdm
import json
from sklearn.metrics import classification_report

# --- 模型和数据集定义 (保持不变) ---
class BiLSTM_CRF(nn.Module):
    # ... 此处代码保持不变 ...
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, dropout):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True, dropout=dropout)
        self.hidden2tag = nn.Linear(hidden_dim, len(tag_to_ix))
        self.crf = CRF(len(tag_to_ix), batch_first=True)

    def forward(self, x):
        embedding = self.embedding(x)
        outputs, _ = self.lstm(embedding)
        feats = self.hidden2tag(outputs)
        return feats

class NERDataset(Dataset):
    # ... 此处代码保持不变 ...
    def __init__(self, data, word_to_ix, tag_to_ix):
        self.sentences = [d['tokens'] for d in data]
        self.tags = [d['tags'] for d in data]
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        sentence_in = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w in sentence]
        tags_in = [self.tag_to_ix[t] for t in tags]
        return torch.tensor(sentence_in, dtype=torch.long), torch.tensor(tags_in, dtype=torch.long)

def collate_fn(batch):
    # ... 此处代码保持不变 ...
    sentences, tags = zip(*batch)
    padded_sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=0)
    return padded_sentences, padded_tags

# --- 评估函数 (保持不变) ---
def _evaluate_model(model, test_loader, ix_to_tag):
    # ... 此处代码保持不变 ...
    print("\n--- 开始评估模型性能 ---")
    model.eval()
    all_preds, all_tags = [], []
    with torch.no_grad():
        for sentence, tags in test_loader:
            emission = model(sentence)
            preds = model.crf.decode(emission, mask=sentence.ne(0))
            
            for p_seq, t_seq in zip(preds, tags):
                all_preds.extend(p_seq)
                all_tags.extend(t_seq[:len(p_seq)].tolist())

    pred_labels = [ix_to_tag.get(p, 'O') for p in all_preds]
    true_labels = [ix_to_tag.get(t, 'O') for t in all_tags]
    
    entity_preds = [p for p, t in zip(pred_labels, true_labels) if t != 'O']
    entity_trues = [t for t in true_labels if t != 'O']
    
    if len(entity_trues) == 0:
        print("测试集中未发现实体标签，无法计算F1分数。")
        return

    print("\n--- 性能报告 (实体标签) ---")
    report = classification_report(entity_trues, entity_preds, zero_division=0)
    print(report)

# --- 训练器主函数 (已升级) ---
def _train_ner_model(task_name: str, config_module: Type):
    print(f"\n{'='*25} 开始培育NER模型: {task_name} {'='*25}")
    
    processed_data_dir = config_module.DATA_DIR / "processed"
    models_dir = config_module.MODELS_DIR # 获取模型目录
    task_config = config_module.TASKS[task_name]
    trainer_config = config_module.TRAINER_CONFIG
    tag_to_ix = task_config["tag_map"]
    ix_to_tag = {i: t for t, i in tag_to_ix.items()}
    
    train_data = pd.read_json(processed_data_dir / f"{task_name}_train.jsonl", lines=True).to_dict('records')
    test_data = pd.read_json(processed_data_dir / f"{task_name}_test.jsonl", lines=True).to_dict('records')
    
    all_words = {word for sent in [d['tokens'] for d in train_data] for word in sent}
    word_to_ix = {word: i + 1 for i, word in enumerate(all_words)}
    word_to_ix["<PAD>"] = 0
    word_to_ix["<UNK>"] = len(word_to_ix)
    vocab_size = len(word_to_ix)

    model = BiLSTM_CRF(vocab_size, tag_to_ix, trainer_config["EMBEDDING_DIM"], trainer_config["HIDDEN_DIM"], trainer_config["DROPOUT"])
    optimizer = optim.Adam(model.parameters(), lr=trainer_config["LEARNING_RATE"])
    
    train_dataset = NERDataset(train_data, word_to_ix, tag_to_ix)
    test_dataset = NERDataset(test_data, word_to_ix, tag_to_ix)
    train_loader = DataLoader(train_dataset, batch_size=trainer_config["BATCH_SIZE"], shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=trainer_config["BATCH_SIZE"], collate_fn=collate_fn)
    
    print("--- 开始训练 ---")
    for epoch in range(trainer_config["EPOCHS"]):
        model.train()
        total_loss = 0
        for sentence, tags in tqdm(train_loader, desc=f"Epoch {epoch+1}/{trainer_config['EPOCHS']}"):
            model.zero_grad()
            emission = model(sentence)
            mask = sentence.ne(word_to_ix["<PAD>"])
            loss = -model.crf(emission, tags, mask=mask, reduction='mean')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} 完成, 平均损失: {total_loss / len(train_loader):.4f}")

    _evaluate_model(model, test_loader, ix_to_tag)
    
    # ======================================================================
    # === 编译器指示修正 (V6) ===
    # 错误原因: 在训练器中直接导出JSON，违反了模块化和单一职责原则。
    # 解决方案: 只导出ONNX模型和临时的JSON文件，将生成最终.bin文件的任务交给exporter。
    print("\n--- 开始导出ONNX模型及临时预处理器数据 ---")
    onnx_path = models_dir / f"{task_name}.onnx"
    dummy_input = torch.randint(0, len(word_to_ix), (1, 20), dtype=torch.long)
    torch.onnx.export(model, dummy_input, onnx_path,
                      input_names=['input'], output_names=['output'],
                      opset_version=11,
                      dynamic_axes={'input': {0: 'batch_size', 1: 'sequence_length'},
                                    'output': {0: 'batch_size', 1: 'sequence_length'}})
    print(f"✅ ONNX模型已导出至: {onnx_path}")

    preprocessor_data = {"word_to_ix": word_to_ix, "tag_to_ix": tag_to_ix}
    json_path = models_dir / f"{task_name}_preprocessor.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessor_data, f, ensure_ascii=False, indent=4)
    print(f"✅ 临时预处理器数据 (JSON) 已导出至: {json_path}")
    # ======================================================================

    print(f"\n{'='*25} 模型 {task_name} 培育完毕 {'='*25}")

def run(config_module: Type):
    models_dir = config_module.MODELS_DIR
    models_dir.mkdir(exist_ok=True)
    for task_name in config_module.TASKS.keys():
        _train_ner_model(task_name, config_module)