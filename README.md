# DPO 訓練文件


## Environment

在助教提供的 `requirements.txt` 之外

```bash
pip install trl # for DPO
pip install tensorboard # for loss visualization
```
請在 `conda` 或 `venv` 環境中安裝。

以下為作業所需模型下載指令：

```bash
hf download "Qwen/Qwen3Guard-Gen-0.6B" --local-dir ./models/guard
hf download "theblackcat102/Qwen3-1.7B-Usefulness-Judge" --local-dir ./models/usefulness
hf download "unsloth/Llama-3.2-3B-Instruct" --local-dir ./models/chat
```

訓練時模型會去使用這些模型。此外也需要下載資料：

```bash
hf download "theblackcat102/ADL_Final_25W_part1_with_cost" --repo-type dataset --local-dir ./data
```

下載資料後在 `data/` 底下找到對應的 `parquet` 檔 (`data/data/test-00000-of-00001.parquet`)，把路徑放到 `config/train.json` 的 `"data"` 底下的 `"file"` 欄位或移動直接該檔案。

## Drive Usage
大型檔案如 checkpoints 或 results 請放在 [ **這裡** ](https://drive.google.com/drive/folders/1lGMseEGDRtiRAip4nqwmG6-fBkLTt0Ys?usp=sharing) 。已開啟編輯權限，注意請勿誤刪他人檔案。  
檔案結構如下，請依格式修改檔名以方便閱讀：
```bash
CVPDL FINAL/
│
├── reward model/
│   # rewriter score & sampler NUM_RETURN_SEQUENCES
│   ├── RewriteScore0.78_SamplerNumreturnsequences24/
│   └── ...
├── sampler result/
│   # rewriter score & sampler NUM_RETURN_SEQUENCES
│   ├── RewriteScore0.78_SamplerNumreturnsequences24/
│   └── ...
└── training checkpoint + inferenced prompt/
    # you should put both rewrite checkpoint + prompt here
    # trainer NUM_RETURN_SEQUENCES (optional: new methods, version tag)
    ├── TrainerNumreturnsequences8_UseAutored_ver3/
    │   ├── rewrite/
    │   └── prompts_ADL_Final_25W_part1_with_cost.jsonl
    └── ...
```

## GPU Optimization
`eval` 相關的許多操作會呼叫 `judge()` 生成句子，這非常費時，因此以下檔案中改為呼叫 `batch_judge()` 函式，設有 `GPU batch` 相關常數，請依自己的能力進行調整 `config` 和以下參數。注意，請 ***不要*** 在同一張 GPU 上跑兩個 process 。  
( 以下為適用於 GPU VRAM 48GB 之設定，若爆 VRAM 可以再調小 )
- `run_eval_progress.py` ： `BATCH_SIZE = 16`
- `src/train.py` ： `NUM_RETURN_SEQUENCES = 8`、`GENERATE_BATCH_SIZE = 8`
- `src/sampler.py` ： `NUM_RETURN_SEQUENCES = 24`


## Training

使用以下指令進行訓練：
```bash
accelerate launch --num_processes 4 src/train.py --config config/train.json
```

`--num_processes` 請換成有的 GPU 數量，而可以更改 `config/train.json` 中的超參數，經測試 $10^{-5}$ 應該是不錯的 LR。此外建議在開始訓練之前，先用 debug mode 試跑一次 (將 `config/train.json` 底下的 `"train"` 底下的 `"debug"` 設為 `true` )。

訓練的流程如下，每個 Iteration，會先從當前模型抽樣一些輸出，並拿取進行 safety 以及 relevance 分數計算，之後會判斷有沒有超過一段時間沒有得到更好的分數。接下來會拿生成的資料得到一些 preference pairs，丟到 DPOTrainer 中進行訓練。在 Trainer 的訓練中也會進行 early stopping。


## Evaluation

將訓練得到的模型複製到 `models/rewrite`：

```bash
cp -r checkpoints/exp{run} models/rewrite
```

接著就可以調整 `src/algorithms.py` 並測試結果： 
```bash
python run_inference_progress.py
python run_eval_progress.py
```

## Reward model (testing)

### Training

將前面得到 rewrite model 放到 `models/rewrite` 下，並且執行：
```bash
accelerate launch --num_processes 4 src/sampler.py [--debug]
accelerate launch --num_processes 4 src/reward.py [--config config/reward.json]
```

`src/sampler.py` 只需要執行一次即可。



