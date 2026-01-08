# Custom OpenCompass Extension

This repository is a fork of [OpenCompass](https://github.com/open-compass/opencompass), extended with **custom model configurations** and **dataset settings** for evaluation experiments.

We add support for additional models under:

```
opencompass/configs/models/a_test_models_instruct/
```

and custom datasets settings under:

```
opencompass/configs/datasets/
```

---

## 🛠️ Installation

Please follow the official [OpenCompass installation guide](https://doc.opencompass.org.cn/get_started/installation.html) to set up the environment.

The vllm we used is `vllm==0.6.1.post1`, it maybe different from the latest opencompass version.

> You may also refer to the customized version of the original README provided here as `README-opencompass.md`.

---

## 🚀 Usage

### 1. Dataset Preparation

Before running any evaluation, please download the necessary datasets according to the [OpenCompass documentation](https://doc.opencompass.org.cn/get_started/installation.html#dataset-preparation).

### 2. Model Path Configuration

After downloading the required models, **update the local model path** in the configuration files under:

```
opencompass/configs/models/a_test_models_instruct/
```

For example:

```python
models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr="gemma-2-2b-instruct-vllm",
        path="/datas/models/gemma2-2B-it",  # 🔧 Replace this with the actual local model path
        model_kwargs=dict(
            tensor_parallel_size=4,
            max_model_len=8192,
            gpu_memory_utilization=0.9
        ),
        max_out_len=4096,
        max_seq_len=8192,
        batch_size=10000,
        generation_kwargs=dict(temperature=0),
        run_cfg=dict(num_gpus=4),
        meta_template=meta_template
    )
]
```

### 3. Running Evaluations

You may run evaluations using the same commands provided in the official documentation. Additionally, this repo includes **custom evaluation scripts** in:

```
opencompass/eval_scripts/run_large_4gpus.sh
```

Feel free to explore and adapt them for your experiments.

---

## 📚 Notes

- Custom dataset configuration files are provided under `opencompass/configs/datasets/`, primarily targeting the **GSM8K** and **MATH** benchmarks.

- To support systematic exploration of different **prompting settings**, the configurations are organized as follows:
  - `a_gsm8k/` for GSM8K
  - `a_math/` for MATH

    These configuration files correspond to the prompting setups used in our paper and are differentiated by their filenames. To run a specific setup, please update the dataset configuration filenames accordingly in your script.

- `gsm8k_postprocess_fixed()` is a customized post-processing function for GSM8K implemented in `opencompass/opencompass/datasets/gsm8k.py`. It is introduced to mitigate evaluation bias present in the original post-processing logic.

  - Please ensure you use this function intentionally in the configuration files. For example:

    ```python
    gsm8k_eval_cfg = dict(
        evaluator=dict(type=Gsm8kEvaluator),
        # Use gsm8k_postprocess_fixed() to reduce evaluation bias.
        # The original gsm8k_postprocess() may introduce bias; switch between them
        # when you need a controlled comparison under the same prompting strategy.
        pred_postprocessor=dict(type=gsm8k_postprocess_fixed),
        dataset_postprocessor=dict(type=gsm8k_dataset_postprocess),
    )
    ```
---

## 📎 References

* [OpenCompass GitHub Repository](https://github.com/open-compass/opencompass)
* Local documentation: `README-opencompass.md`


