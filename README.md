# DPO 訓練文件


## 環境

在助教提供的 `requirements.txt` 之外

```bash
pip install trl # for DPO
pip install tensorboard # for loss visualization
```
請在 `conda` 或 `venv` 環境中安裝。

將以下模型先下載下來：

```bash
huggingface-cli download "Qwen/Qwen3Guard-Gen-0.6B" --local-dir ./models/guard
huggingface-cli download "theblackcat102/Qwen3-1.7B-Usefulness-Judge" --local-dir ./models/usefulness
huggingface-cli download "unsloth/Llama-3.2-3B-Instruct" --local-dir ./models/chat
```

在訓練的值時候模型會去使用這些模型。此外也需要下載資料：

```bash
huggingface-cli download "theblackcat102/ADL_Final_25W_part1_with_cost" --repo-type dataset --local-dir ./data
```

在 `data/` 底下找到對應的 `parquet` 檔(`*.parquet`)，把路徑放到 `config/train.json` 的 `"data"` 底下的 `"file"` 欄位。

## 訓練

使用以下指令進行訓練：
```bash
accelerate launch --num_processes 4 src/train.py --config config/train.json
```

`--num_processes` 可以換成有的 GPU 數量，而可以更改 `config/train.json` 中的超參數，經測試 $10^{-5}$ 應該是不錯的 LR。此外建議在開始訓練之前，先用 debug mode 試跑一次，將 `config/train.json` 底下的 `"train"` 底下的 `"debug"` 設為 `true` 即可。

訓練的流程如下，每個 Iteration，會先從當前模型抽樣一些輸出，並拿取進行 safety 以及 relevance 分數計算，之後會判斷有沒有超過一段時間沒有得到更好的分數。接下來會拿生成的資料得到一些 preference pairs，丟到 DPOTrainer 中進行訓練。在 Trainer 的訓練中也會進行 early stopping。


## Evaluation

將訓練得到的模型複製到 `models/rewrite`：

```bash
cp -r checkpoints/exp{run} models/rewrite
```

接著就可以使用 
```bash
python run_inference.py
python run_eval.py
```



