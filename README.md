# fei-einops
This package mimics the idea in [Einops](https://github.com/arogozhnikov/einops) library

For example:
- `rearrange t "h w c -> c h w" []` will transpose the tensor from HWC to CHW.
- `rearrange t "b n c -> b (n c) []"` will reshape the tensor by merging up the last 2 axes.
- `rearrange t "(b n) c -> b n c" [#b .== 2]` will reshape the tensor by decompose the 1st axis into 2, where the new axis `n` is calculated automatically.
- ...

In most cases, the tensor can be `SymbolHandle`, `NDArrayHandle`, `NDArray` for mxnet.
