# 使用 OmniDocBench 对 Gemini-2.0-flash-exp 进行评测

## 安装

```bash
# 克隆仓库
git clone https://github.com/ninehills/OmniDocBench.git
cd OmniDocBench/

# 创建虚拟环境
conda create -n omnidocbench python=3.10
conda activate omnidocbench

# 安装依赖
pip install -r requirements.txt
```

## 测试

根据 `demo_data` 中的数据跑一个 validation：

1. predict 数据： `demo_data/end2end` 目录下
2. ground truth 数据： `demo_data/omnidocbench_demo` 目录下

### 1. md2md 方式

md2md 方式，目前正式数据集已经不适用这种方式，但是比较好构造 ground truth 数据。

1. predict 数据格式：markdown
2. ground truth 数据格式：markdown

add `./configs/md2md_wo_info.yaml` 文件，内容如下：

```yaml
end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist
        - METEOR
        - BLEU
    display_formula:
      metric:
        - Edit_dist
        - CDM
    table:
      metric:
        - TEDS
        - Edit_dist
    reading_order:
      metric:
        - Edit_dist
  dataset:
    dataset_name: md2md_dataset
    ground_truth:
      data_path: ./demo_data/omnidocbench_demo/mds
    prediction:
      data_path: ./demo_data/end2end
    match_method: quick_match
```

运行：

```bash
$ python pdf_validation.py --config ./configs/md2md_wo_info.yaml
【text_block】
Edit_dist:
------------  --------
ALL_page_avg  0.359956
------------  --------
====================================================================================================
METEOR:
---  --------
all  0.114545
---  --------
====================================================================================================
BLEU:
---  --------
all  0.269941
---  --------
====================================================================================================
【display_formula】
Edit_dist:
------------  --------
ALL_page_avg  0.474891
------------  --------
====================================================================================================
【table】
TEDS:
---  --------
all  0.783813
---  --------
====================================================================================================
TEDS_structure_only:
---  --------
all  0.911589
---  --------
====================================================================================================
Edit_dist:
------------  --------
ALL_page_avg  0.202719
------------  --------
====================================================================================================
【reading_order】
Edit_dist:
------------  --------
ALL_page_avg  0.250363
------------  --------
====================================================================================================
```

### 2. end2end 方式

end2end 方式，目前正式数据集支持，需要用此种方式整体进行评估。

1. predict 数据格式：markdown
2. ground truth 数据格式：json

```bash
$ python pdf_validation.py --config ./configs/end2end.yaml
【text_block】
Edit_dist:
------------  --------
ALL_page_avg  0.356126
...
```

输出会保存在 `./result` 目录下，以 predict 目录名称命名。

数据和 md2md 会略有不同，但是整体结果会差不多，而且更加精细。

### 3. 分析数据

使用 `./tools/generate_result_tables.ipynb` 进行数据分析，并计算 overall 结果。

```
overall_EN	overall_CH
0.194	0.354
```

## 下载正式数据集

下载正式数据集：

```bash
$ huggingface-cli download --repo-type dataset opendatalab/OmniDocBench --local-dir check_data
```

## 使用自定义模型进行评测

我们用最新的 gemini-2.0-flash 以及 doc2x 模型进行评测。

1. 实现模型推理脚本。

我们以 `tools/model_infer/Qwen2VL_img2md.py` 为模版，使用 OpenRouter 服务 / Gemini 服务 + OpenAI 客户端进行模型推理。

```bash
$ pip install openai python-dotenv google-genai==0.2.2
$ vim .env
# 增加 OPENAI_API_KEY / OPENAI_BASE_URL（openrouter）或者 GOOGLE_API_KEY (gemini)
$ python tools/model_infer/openai_img2md.py --sample --model "google/gemini-2.0-flash-exp:free" --input-dir ./demo_data/omnidocbench_demo/images --output-dir ./output/gemini-2.0-flash-end2end --provider openai
$ python tools/model_infer/openai_img2md.py --sample --model "gemini-2.0-flash-exp" --input-dir ./demo_data/omnidocbench_demo/images --output-dir ./output/gemini-2.0-flash-end2end --provider gemini
```

2. 在 `demo_data` 中跑通推理 + 评测。

```bash
$ python tools/model_infer/openai_img2md.py --model "gemini-2.0-flash-exp" --input-dir ./demo_data/omnidocbench_demo/images --output-dir ./output/gemini-2.0-flash-end2end --provider gemini

# benchmark
$ cat > ./configs/gemini_end2end.yaml <<EOF
end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist
        - BLEU
        - METEOR
    display_formula:
      metric:
        - Edit_dist
        - CDM
    table:
      metric:
        - TEDS
        - Edit_dist
    reading_order:
      metric:
        - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./demo_data/omnidocbench_demo/OmniDocBench_demo.json
    prediction:
      data_path: ./output/gemini-2.0-flash-end2end
    match_method: quick_match
    # filter:
    #   language: english
EOF
$ python pdf_validation.py --config ./configs/gemini_end2end.yaml
```

3. 在 `check_data` 中跑通正式数据集的推理 + 评测。

推理：

```bash
$ python tools/model_infer/openai_img2md.py --model "gemini-2.0-flash-exp" --input-dir ./check_data/images --output-dir ./output/check/gemini-2.0-flash-e2e --provider gemini --qps 0.1
```

评测：

```bash
# benchmark
$ cat > ./configs/gemini_end2end_check.yaml <<EOF
end2end_eval:
  metrics:
    text_block:
      metric:
        - Edit_dist
        - BLEU
        - METEOR
    display_formula:
      metric:
        - Edit_dist
        - CDM
    table:
      metric:
        - TEDS
        - Edit_dist
    reading_order:
      metric:
        - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: ./check_data/OmniDocBench.json
    prediction:
      data_path: ./output/check/gemini-2.0-flash-e2e
    match_method: quick_match
    # filter:
    #   language: english
EOF
$ python pdf_validation.py --config ./configs/gemini_end2end_check.yaml
```

## 结果分析

使用 `./tools/generate_result_tables.ipynb` 进行结果分析。


<table style="width: 92%; border-collapse: collapse; margin: 0 auto;">
  <caption>Comprehensive evaluation of document parsing algorithms on OmniDocBench: performance metrics for text, formula, table, and reading order extraction, with overall scores derived from ground truth comparisons.</caption>
  <thead>
    <tr>
      <th rowspan="2">Method Type</th>
      <th rowspan="2">Methods</th>
      <th colspan="2">Text<sup>Edit</sup>↓</th>
      <th colspan="2">Formula<sup>Edit</sup>↓</th>
      <th colspan="2">Formula<sup>CDM</sup>↑</th>
      <th colspan="2">Table<sup>TEDS</sup>↑</th>
      <th colspan="2">Table<sup>Edit</sup>↓</th>
      <th colspan="2">Read Order<sup>Edit</sup>↓</th>
      <th colspan="2">Overall<sup>Edit</sup>↓</th>
    </tr>
    <tr>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
      <th>EN</th>
      <th>ZH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Pipeline Tools</td>
      <td>MinerU-0.9.3</td>
      <td><strong>0.058</strong></td>
      <td><strong>0.211</strong></td>
      <td><strong>0.278</strong></td>
      <td>0.577</td>
      <td>66.9</td>
      <td>49.5</td>
      <td><strong>79.4</strong></td>
      <td>62.7</td>
      <td><strong>0.305</strong></td>
      <td><u>0.461</u></td>
      <td><strong>0.079</strong></td>
      <td>0.288</td>
      <td><strong>0.180</strong></td>
      <td><u>0.384</u></td>
    </tr>
    <tr>
      <td>Marker-0.2.17</td>
      <td>0.141</td>
      <td>0.303</td>
      <td>0.667</td>
      <td>0.868</td>
      <td>18.4</td>
      <td>12.7</td>
      <td>54.0</td>
      <td>45.8</td>
      <td>0.718</td>
      <td>0.763</td>
      <td>0.138</td>
      <td>0.306</td>
      <td>0.416</td>
      <td>0.560</td>
    </tr>
    <tr>
      <td>Mathpix</td>
      <td><u>0.101</u></td>
      <td>0.358</td>
      <td><u>0.306</u></td>
      <td><strong>0.454</strong></td>
      <td>71.4</td>
      <td><strong>72.7</strong></td>
      <td><u>77.9</u></td>
      <td><strong>68.2</strong></td>
      <td><u>0.322</u></td>
      <td><strong>0.416</strong></td>
      <td><u>0.105</u></td>
      <td>0.275</td>
      <td><u>0.209</u></td>
      <td><strong>0.376</strong></td>
    </tr>
    <tr>
      <td rowspan="2">Expert VLMs</td>
      <td>GOT-OCR</td>
      <td>0.187</td>
      <td>0.315</td>
      <td>0.360</td>
      <td><u>0.528</u></td>
      <td><strong>81.8</strong></td>
      <td>51.4</td>
      <td>53.5</td>
      <td>48.0</td>
      <td>0.521</td>
      <td>0.594</td>
      <td>0.141</td>
      <td>0.28</td>
      <td>0.302</td>
      <td>0.429</td>
    </tr>
    <tr>
      <td>Nougat</td>
      <td>0.365</td>
      <td>0.998</td>
      <td>0.488</td>
      <td>0.941</td>
      <td>17.4</td>
      <td>16.9</td>
      <td>40.3</td>
      <td>0.0</td>
      <td>0.622</td>
      <td>1.000</td>
      <td>0.382</td>
      <td>0.954</td>
      <td>0.464</td>
      <td>0.973</td>
    </tr>
    <tr>
      <td rowspan="4">General VLMs</td>
      <td>GPT4o</td>
      <td>0.144</td>
      <td>0.409</td>
      <td>0.425</td>
      <td>0.606</td>
      <td><u>76.4</u></td>
      <td>48.2</td>
      <td>72.75</td>
      <td>63.7</td>
      <td>0.363</td>
      <td>0.474</td>
      <td>0.128</td>
      <td>0.251</td>
      <td>0.265</td>
      <td>0.435</td>
    </tr>
    <tr>
      <td>Qwen2-VL-72B</td>
      <td>0.252</td>
      <td><u>0.251</u></td>
      <td>0.468</td>
      <td>0.572</td>
      <td>54.9</td>
      <td><u>60.9</u></td>
      <td>59.9</td>
      <td><u>66.8</u></td>
      <td>0.591</td>
      <td>0.587</td>
      <td>0.255</td>
      <td><strong>0.223</strong></td>
      <td>0.392</td>
      <td>0.408</td>
    </tr>
    <tr>
      <td>InternVL2-Llama3-76B</td>
      <td>0.353</td>
      <td>0.290</td>
      <td>0.543</td>
      <td>0.701</td>
      <td>69.8</td>
      <td>49.6</td>
      <td>63.8</td>
      <td>61.1</td>
      <td>0.616</td>
      <td>0.638</td>
      <td>0.317</td>
      <td><u>0.228</u></td>
      <td>0.457</td>
      <td>0.464</td>
    </tr>
    <tr>
      <td>Gemini-2.0-flash-exp</td>
      <td>0.134</td>
      <td>0.223</td>
      <td>0.424</td>
      <td>0.493</td>
      <td>-</td>
      <td>-</td>
      <td>77.314</td>
      <td>71.632</td>
      <td>0.212</td>
      <td>0.271</td>
      <td>0.077</td>
      <td>0.159</td>
      <td>0.212</td>
      <td>0.286</td>
    </tr>
  </tbody>
</table>


注：公式的 CDM 为空是因为其需要安装 [CDM](https://github.com/opendatalab/UniMERNet/blob/main/cdm/README-CN.md) 评测工具后，使用 `results/xxx_formula.json` 文件进行评测。