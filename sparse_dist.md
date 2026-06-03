# Qwen3-30B-A3B SparseAttention 蒸馏训练实现计划

## 总体思路

- Teacher (A): 原始 Qwen3-30B-A3B，使用标准 `SelfAttention`（Megatron-LM `DotProductAttention` / TE attention）。
- Student (B): 同结构 Qwen3-30B-A3B，将每层 `SelfAttention` 内的 `core_attention` 替换为 `FlashMaskAttention`（先用 SDPA + 自定义 mask 占位，后续可替换为真正的稀疏 kernel）。
- 训练时同时前向 A、B；通过 forward hook 抓取每层 attention 在 `linear_proj` 之前的 core 输出（shape `[s, b, h]`，h = np * hn），对每层做 KL loss，与原 SFT CE loss 相加，仅反传 student。

## 架构图

```
   batch ──┬─► Teacher (no_grad, eval) ──hooks──► [t_attn_out_l] (per layer)
           │
           └─► Student (FlashMaskAttn) ─hooks──► [s_attn_out_l] (per layer)
                       │                                    │
                       ▼                                    ▼
                   LM CE loss ◄──────── + α * Σ KL(s_l || t_l) ◄──── kl_per_layer
                       │
                       ▼
                   backward (student only)
```

## 要修改 / 新增的文件

### 新增

1. `src/megatron/bridge/models/qwen/sparse_attention.py`
   - `class FlashMaskAttention(MegatronModule)`：与 `megatron.core.transformer.dot_product_attention.DotProductAttention` 接口一致 —— `__init__(config, layer_number, attn_mask_type, attention_type)`，`forward(query, key, value, attention_mask, attn_mask_type=None, packed_seq_params=None)`。
   - 内部使用 `torch.nn.functional.scaled_dot_product_attention` + 自定义 sparse bool mask（占位 = causal），返回 `[s, b, h]` 与 Megatron 约定一致。
   - `def build_flash_mask(seq_len, device, dtype, sparse_pattern=None)`：构造 mask（占位为 causal；预留 column-wise 接口）。

2. `src/megatron/bridge/models/qwen/qwen3_swap_attention.py`
   - `def swap_to_flashmask(model)`：遍历 `model.decoder.layers[i].self_attention`，将 `core_attention` 替换为 `FlashMaskAttention(config=...)`。
   - `def register_attn_hooks(model, collector)`：在每个 `self_attention.core_attention` 上注册 forward hook，将输出 push 到 collector。

3. `src/megatron/bridge/training/sparse_distill.py`
   - `class AttnOutputCollector`：缓存 hook 输出列表；`clear()` / `get()`。
   - `def kl_per_layer(student_outs, teacher_outs, loss_mask, temperature=1.0) -> Tensor`：对每对 `[s,b,h]` 在最后一维做 `log_softmax/softmax`，`F.kl_div(reduction='none')`，按 `loss_mask` 做加权 token 平均，跨层求和。
   - `def build_teacher_model(cfg) -> list[GPTModel]`：基于 cfg 复制一份 model provider，加载 HF 权重，`eval()` + `requires_grad_(False)`。
   - `class SparseDistillForwardStep`：可作为 `forward_step_func` 传给 `pretrain()`。`__init__(teacher_models, alpha, temperature)`；`__call__(state, data_iterator, model)` 复用 `_forward_step_common` 拿到 batch 与 student logits，并在 teacher 上跑 `no_grad` forward 抓 attention。
   - `def loss_fn_with_kl(output_tensor, loss_mask, student_outs, teacher_outs, alpha, temperature, ...)`：组合 LM loss + α·KL。

4. `examples/distillation/qwen3/sparse_distill_qwen3_30b.py`
   - 入口脚本：调用 `qwen3_30b_a3b_sft_config()`，构造 student（`load_weights=True` + `swap_to_flashmask`）；构造 teacher provider 一份；调用 `pretrain(cfg, forward_step_func=SparseDistillForwardStep(...))`。
   - 参考 `examples/distillation/llama/distill_llama32_3b-1b.py`。

### 修改

- `src/megatron/bridge/recipes/qwen/qwen3_moe.py`
  - 新增 `qwen3_30b_a3b_sparse_distill_config()`：基于 `qwen3_30b_a3b_sft_config()` 复制；将 `cfg.model.transformer_impl = "local"`（确保 hook 能拿到 `core_attention` python 对象，避免 TE 的 fused 实现）；附加属性 `cfg.distill = SimpleNamespace(alpha=1.0, temperature=1.0)`。

- 新建 `train_qwen35b_sparse_distill.sh`（在 repo 根目录，与 `train_qwen35b.sh` 并列）：入口改为 `examples/distillation/qwen3/sparse_distill_qwen3_30b.py`，TP/PP/EP 与原脚本一致。

## 关键设计点

1. **KL 对齐对象**：core attention 输出（o_proj 之前），shape `[s, b, h]`，h = num_heads * head_dim。在最后一维上做 softmax/log_softmax 以获得分布形式后再算 KL。
2. **FlashMaskAttention 占位**：`F.scaled_dot_product_attention(q,k,v, attn_mask=build_flash_mask(...), is_causal=False)`。q/k/v 形状 `[s, b, np, hn]` → 转到 `[b, np, s, hn]` 调用 → 转回 `[s, b, h]`。
3. **Teacher 不反传**：`requires_grad_(False)`、`eval()`。
4. **Hook 时机**：`forward_hook(module, inp, out)`，其中 `out` 是 `core_attention` 的 dense 输出 `[s, b, h]`，即 o_proj 输入；这是希望对齐的 "attention 输出"。
5. **PP=1 先行**：当前 30B-A3B SFT recipe 里 PP=2，蒸馏 recipe 中改为 PP=1（teacher/student 各 4 GPU），配合 EP 与 TP 分配。
6. **MoE/dispatcher 无影响**：attention 在 dispatcher 之前。

## 待用户进一步确认（不阻塞实现）

- KL normalize 维度：默认 last-dim (`h`)。
- α、T 默认值：1.0 / 1.0。
- 是否保留 LM CE loss：默认保留（α 控制 KL 权重）。

## 后续可选增强

- SDPA 占位 → 真 FlashMask kernel（Triton/CUDA）。
- 支持 PP>1：每 stage 内独立做 KL 后 all-reduce；最后 stage 输出总 loss。
- 替代 MSE / 余弦距离 loss。
