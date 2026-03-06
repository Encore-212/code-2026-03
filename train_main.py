import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import os
import logging
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from model import Audio
from VisualExperiment import save_roc_arrays, model_complexity_simple, printResult
import copy
from CosineSimilarity import compute_dualstream_table
import glob
from paired_test import save_metrics

ModelClass = Audio
MODEL_NAME = 'MODEL_NAME'
DATASET_NAME = ''
THOP_AVAILABLE = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def make_paths(fold_idx: int):
    # ========= ROC数据保存 单独设置位置 ==========
    roc_save_dir = 'roc_save_dir'
    roc_npz_path = os.path.join(roc_save_dir, "roc_cache", DATASET_NAME, MODEL_NAME)
    # == == == == ==  == == == == ==
    model_dir = os.path.join(BASE_DIR,"model_dict", "Net", MODEL_NAME)
    model_path = os.path.join(model_dir, f"model_fold{fold_idx}.pth")
    metrics_path = os.path.join(BASE_DIR,"model_dict", "Net","Pvalue_metrics", MODEL_NAME)
    # 保证目录存在
    os.makedirs(roc_npz_path, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)
    return roc_npz_path, model_path, metrics_path

class CMDCMFCCAllDataset(Dataset):
    def __init__(self, data_list, feature_dir):
        self.data_entries = []
        for sample in data_list:
            ID = sample["ID"]
            label = sample["label"]
            pattern = os.path.join(feature_dir, f"{ID}_Q*.npy")
            matched_files = glob.glob(pattern)
            for file_path in matched_files:
                self.data_entries.append((file_path, label))
    def __len__(self):
        return len(self.data_entries)
    def __getitem__(self, idx):
        file_path, label = self.data_entries[idx]
        features = np.load(file_path)  # shape: (1, 40, 890)
        # 标准化
        features = (features - np.mean(features)) / (np.std(features) + 1e-6)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def valid_with_scores(model, device, valid_loader):
    model.eval()
    all_predicted = []
    all_trues = []
    all_scores = []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if outputs.dim() == 2 and outputs.size(1) == 2:
                probs = torch.softmax(outputs, dim=1)  # (B,2)
                y_score = probs[:, 1]  # (B,)
                y_pred = torch.argmax(probs, dim=1)  # (B,)
            elif outputs.dim() == 1 or (outputs.dim() == 2 and outputs.size(1) == 1):
                logits = outputs.view(-1)
                y_score = torch.sigmoid(logits)  # (B,)
                y_pred = (y_score >= 0.5).long()  # (B,)
            else:
                raise ValueError(f"不支持的输出形状: {tuple(outputs.shape)}")
            all_predicted.append(y_pred.cpu().numpy())
            all_trues.append(targets.cpu().numpy())
            all_scores.append(y_score.cpu().numpy())

    all_predicted = np.concatenate(all_predicted)
    all_trues = np.concatenate(all_trues)
    all_scores = np.concatenate(all_scores)
    return all_trues, all_predicted, all_scores

def run_five_fold(random_seed, run_time, config: dict, json_path, feature_dir,
                  batch_size=32, num_epochs=200, lr=0.00141, clip=1.0, optim_name='Adam'):
    with open(json_path, 'r') as f:
        folds = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =================配置log文件=====================
    log_dir =os.path.join(BASE_DIR,"log", "Net", MODEL_NAME, run_time)
    os.makedirs(log_dir, exist_ok=True)

    for fold_index, (fold_name, split) in enumerate(folds.items(), start=1):
        # =================配置log文件=====================
        log_file = f"{log_dir}/fold_index{fold_index}.log"
        # 清空 handler（Python 3.7 必须）
        for h in logging.root.handlers[:]:
            logging.root.removeHandler(h)
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s"
        )
        logging.info(f"Start training, fold_index={fold_index}")
        # ======================================
        train_dataset = CMDCMFCCAllDataset(split['train'], feature_dir)
        test_dataset = CMDCMFCCAllDataset(split['test'], feature_dir)
        g = torch.Generator()
        g.manual_seed(random_seed)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, generator=g)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, generator=g)

        model = ModelClass(config).to(device)
        logging.info(f"\n>>> 使用模型类：{ModelClass.__name__}\n>>> 来源模块：{ModelClass.__module__}")

        # =============计算模型复杂度（在训练前）=====================
        # 获取一个真实的batch
        sample_input, _ = next(iter(train_loader))
        sample_input = sample_input.to(device)
        # 计算复杂度
        if THOP_AVAILABLE:
            try:
                # 用副本统计复杂度，避免污染原模型
                model_for_profile = copy.deepcopy(model).to(device)
                complexity_info = model_complexity_simple(model_for_profile, sample_input, device=device)
            except Exception as e:
                print(f"复杂度计算失败: {e}")
                complexity_info = None

        roc_npz_path, model_path, metrics_path = make_paths(fold_index)
        #  ========================================
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        else:
            all_train_labels = [item['label'] for item in split['train']]
            class_weights = compute_class_weight('balanced', classes=np.unique(all_train_labels), y=all_train_labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = getattr(optim, optim_name)(model.parameters(), lr=lr)
            best_val_acc = 0.0
            for epoch in range(num_epochs):
                model.train()
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    optimizer.zero_grad()
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()

                # 验证并获取预测概率
                model.eval()
                val_trues, val_predicted, _ = valid_with_scores(model, device, test_loader)
                val_acc = accuracy_score(val_trues, val_predicted)

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), model_path)
                    logging.info(f"[Fold {fold_index}] Epoch {epoch}: 保存当前最优模型到 {model_path}")
            # 加载最佳模型以进行最终测试
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
        # ====== 双流相似度统计（Fold级，权重确定后，最终测试前）======
        try:
            dual_df = compute_dualstream_table(
                model=model,
                dataloader=test_loader,
                device=device,
                tau=0.7,
                tag=f"Fold{fold_index}_Stream1_vs_Stream2"
            )
            print(dual_df)

            # 建议：保存到本次实验目录下，并带fold信息，避免被覆盖
            dual_csv = os.path.join(log_dir, f"dualstream_fold{fold_index}.csv")
            dual_df.to_csv(dual_csv, index=False)

            logging.info(f"Dual-stream similarity table saved: {dual_csv}")
            logging.info("\n" + dual_df.to_string(index=False))
        except Exception as e:
            logging.exception(f"Dual-stream similarity table failed: {e}")
        # ============================================================
        # 测试集
        all_trues, all_predicted, all_scores = valid_with_scores(model, device, test_loader)

        # ==============保存roc曲线================
        save_roc_arrays(
            save_dir=roc_npz_path,
            model_tag="roc",
            split_tag=f"fold{fold_index}",
            y_true=all_trues,
            y_score=all_scores ,
            y_pred=all_predicted
        )

        metrics = printResult(all_trues, all_predicted, all_scores, complexity_info, seed=random_seed)
        # ======================= 保存P值=========================
        metrics['model_tag'] = f"Pvalue"
        metrics['fold'] = f"{fold_index}"
        save_metrics(metrics, metrics_path)


def train_main(config: dict, random_seed, run_time):
    # 设置全局随机种子
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    five_fold_json_path = r'five_fold_json_path'
    feature_dir = r'feature_dir'

    model_path = os.path.join(BASE_DIR, 'model_dict', 'Net', MODEL_NAME)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    config['save_model_pth'] = model_path

    results = run_five_fold(
        random_seed,
        run_time,
        config, five_fold_json_path, feature_dir
    )
    return results

