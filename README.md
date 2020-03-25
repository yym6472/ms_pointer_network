# ms_pointer_network

Re-implementation of multi-source pointer network (from the paper [Multi-Source Pointer Network for Product Title Summarization](https://arxiv.org/pdf/1808.06885.pdf)).

## Requirements

- python - 3.7
- pytorch - 1.4.0
- allennlp - 0.9.0

## How to run

### Train model
  ```
  python3 train.py --config_path ./configs/ms_pointer.json --output_dir ./outputs/ms_pointer
  ```
### Test model
  ```
  python3 test.py --output_dir ./outputs/ms_pointer
  ```

## 数据格式

- tsv文件，第一列为source1的tokens，第二列为source2的tokens，第三列为target的tokens
- 每一列的tokens均使用空格分隔
- target中的词必须在两个source中出现至少一次

## TODO

- 目前的实现共用一个词表，是否需要扩展到独立的词表，以解决oov的复制问题？