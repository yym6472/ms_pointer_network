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