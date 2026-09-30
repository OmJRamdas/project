# Simple C++ Inference Engine

Learning project: load an ONNX model and run it on a laptop CPU, written by hand in C++. AI helps with planning and questions only.

## Feasibility

Feasible. A small engine for an MLP and a small CNN (MNIST) is a few weeks of part-time work. Matching onnxruntime's speed or op coverage is not a goal.

- **Format:** ONNX (protobuf). Exporters exist for PyTorch, TF and sklearn.
- **Execution:** ONNX graphs are topologically sorted, so run nodes in order and dispatch on `op_type`.
- **Ops:** about 10-15 cover MNIST-class models (Conv, Gemm, MatMul, Add, Relu, MaxPool, Flatten, Reshape, Softmax, BatchNorm).
- **Reference:** Python `onnxruntime` gives expected outputs for testing.
- **Risks:** op semantics (padding, broadcasting, opsets) and protobuf setup.

**Scope:** CPU, float32, NCHW, static shapes, single-threaded. No GPU, quantization or training.

## Plan

1. CMake project and a Python venv (`onnx`, `onnxruntime`, `numpy`) for test models.
2. `Tensor` (shape + `std::vector<float>`).
3. Parse ONNX into graph, nodes, attributes, initializers (protobuf lib vs. hand-written parser: decide here).
4. Executor with an op registry.
5. Ops: MLP first (Gemm, Add, Relu, Softmax), then CNN (Conv, MaxPool, BatchNorm).
6. Compare against onnxruntime per op and end to end.
7. Optional: benchmark, im2col + GEMM, OpenMP.

## Milestones

- [ ] Print a model's graph
- [ ] MLP matches onnxruntime
- [ ] MNIST CNN matches onnxruntime
