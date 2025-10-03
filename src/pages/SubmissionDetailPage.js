import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { mockSubmissionsByOperator, operatorInfo } from '../data/mockData';

const SubmissionDetailPage = () => {
    const { operatorKey, submissionId } = useParams();
    const navigate = useNavigate();
    const [submission, setSubmission] = useState(null);

    useEffect(() => {
        if (operatorKey && submissionId && mockSubmissionsByOperator[operatorKey]) {
            const found = mockSubmissionsByOperator[operatorKey].find(
                s => s.id === parseInt(submissionId)
            );
            setSubmission(found);
        }
    }, [operatorKey, submissionId]);

    const generateKernelCode = (submission, operatorTitle) => {
        const operatorLower = operatorTitle.toLowerCase().replace(/\s+/g, '_');
        const submissionLower = submission.name.toLowerCase().replace(/[^a-z0-9]/g, '_');
        
        switch (submission.dsl) {
            case 'Triton':
                return `import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def ${submissionLower}_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    High-performance ${operatorTitle} kernel optimized for ${submission.device}
    Achieves ${submission.performance} TFLOPS with ${submission.correctness}% correctness
    """
    # Program ID and grid computation
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block start offsets
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize pointers to A and B
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Main computation loop with memory optimization
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        
        # Perform matrix multiplication with Tensor Core acceleration
        accumulator += tl.dot(a, b)
        
        # Advance pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Convert to output precision and apply optimizations
    c = accumulator.to(tl.float16)

    # Write output with proper masking
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)`;

            case 'CUDA':
                return `#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>

// ${submission.name}: Optimized ${operatorTitle} for ${submission.device}
// Performance: ${submission.performance} TFLOPS
// Memory Bandwidth: ${submission.memoryBW} TB/s

using namespace nvcuda;

// Tensor Core configuration for optimal performance
const int WMMA_M = 16;
const int WMMA_N = 16; 
const int WMMA_K = 16;

// Thread block configuration optimized for ${submission.device}
const int BLOCK_SIZE_M = 128;
const int BLOCK_SIZE_N = 128;
const int BLOCK_SIZE_K = 32;
const int WARP_SIZE = 32;

__global__ void ${submissionLower}_kernel(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K,
    int lda, int ldb, int ldc,
    float alpha, float beta
) {
    // Thread and warp identification
    const int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Shared memory for cooperative loading (optimized for bank conflicts)
    __shared__ half As[BLOCK_SIZE_M * BLOCK_SIZE_K];
    __shared__ half Bs[BLOCK_SIZE_K * BLOCK_SIZE_N];
    
    // Tensor Core fragments for WMMA operations
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Global memory offsets for this thread block
    const int globalRow = blockIdx.y * BLOCK_SIZE_M;
    const int globalCol = blockIdx.x * BLOCK_SIZE_N;

    // Main computation loop over K dimension
    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        // Collaborative loading with coalesced memory access
        const int loadRow = threadIdx.y;
        const int loadCol = threadIdx.x;
        
        // Load A block to shared memory (row-major)
        if (globalRow + loadRow < M && k + loadCol < K) {
            As[loadRow * BLOCK_SIZE_K + loadCol] = A[(globalRow + loadRow) * lda + k + loadCol];
        } else {
            As[loadRow * BLOCK_SIZE_K + loadCol] = __float2half(0.0f);
        }
        
        // Load B block to shared memory (column-major for optimal WMMA)  
        if (k + loadRow < K && globalCol + loadCol < N) {
            Bs[loadRow * BLOCK_SIZE_N + loadCol] = B[(k + loadRow) * ldb + globalCol + loadCol];
        } else {
            Bs[loadRow * BLOCK_SIZE_N + loadCol] = __float2half(0.0f);
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Compute using Tensor Cores with WMMA API
        for (int wk = 0; wk < BLOCK_SIZE_K; wk += WMMA_K) {
            const int warpRow = (warpM % (BLOCK_SIZE_M / WMMA_M)) * WMMA_M;
            const int warpCol = (warpN % (BLOCK_SIZE_N / WMMA_N)) * WMMA_N;
            
            // Load matrix fragments from shared memory
            wmma::load_matrix_sync(a_frag, &As[warpRow * BLOCK_SIZE_K + wk], BLOCK_SIZE_K);
            wmma::load_matrix_sync(b_frag, &Bs[wk * BLOCK_SIZE_N + warpCol], BLOCK_SIZE_N);
            
            // Perform mixed-precision matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        
        __syncthreads();
    }
    
    // Store result back to global memory with alpha/beta scaling
    const int warpRow = (warpM % (BLOCK_SIZE_M / WMMA_M)) * WMMA_M + globalRow;
    const int warpCol = (warpN % (BLOCK_SIZE_N / WMMA_N)) * WMMA_N + globalCol;
    
    if (warpRow < M && warpCol < N) {
        // Load existing C values for beta scaling
        wmma::load_matrix_sync(c_frag, &C[warpRow * ldc + warpCol], ldc, wmma::mem_row_major);
        
        // Apply alpha and beta scaling: C = alpha * A * B + beta * C
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        
        // Store final result to global memory
        wmma::store_matrix_sync(&C[warpRow * ldc + warpCol], c_frag, ldc, wmma::mem_row_major);
    }
}`;

            case 'PyTorch':
                return `import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

class ${submission.name.replace(/[^a-zA-Z0-9]/g, '')}(nn.Module):
    """
    Optimized ${operatorTitle} implementation for ${submission.device}
    Performance: ${submission.performance} TFLOPS
    Efficiency: ${submission.efficiency}%
    """
    
    def __init__(self, input_dim, output_dim, bias=True, precision='${submission.precision}'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.precision = precision
        
        # Initialize parameters with optimal distribution
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter('bias', None)
            
        # Initialize with Xavier/Glorot initialization for stable training
        self.reset_parameters()
        
        # JIT compile custom kernel for maximum performance
        self._compiled_kernel = None
        
    def reset_parameters(self):
        """Initialize parameters for optimal convergence"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    @torch.jit.script_method  
    def fused_forward(self, input: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        """
        JIT-compiled fused forward pass with activation
        Reduces memory bandwidth by fusing operations
        """
        # Apply linear transformation
        output = F.linear(input, self.weight, self.bias)
        
        # Fused activation functions for better performance
        if activation == "relu":
            return F.relu(output, inplace=True)
        elif activation == "gelu":
            return F.gelu(output)
        elif activation == "swish":
            return output * torch.sigmoid(output)
        elif activation == "leaky_relu":
            return F.leaky_relu(output, negative_slope=0.01, inplace=True)
        else:
            return output
    
    def forward(self, input):
        """
        Forward pass with automatic optimization selection
        """
        # Input validation
        if input.dim() != 2:
            raise ValueError(f"Expected 2D input, got {input.dim()}D")
        if input.size(1) != self.input_dim:
            raise ValueError(f"Input dim mismatch: expected {self.input_dim}, got {input.size(1)}")
        
        # Automatic Mixed Precision for ${submission.device}
        if self.precision == 'FP16' and input.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                return F.linear(input, self.weight, self.bias)
        else:
            # Use custom optimized kernel for maximum performance
            return self._optimized_linear(input)
    
    def _optimized_linear(self, input):
        """
        Custom optimized linear operation
        Achieves ${submission.performance} TFLOPS on ${submission.device}
        """
        # Use torch.addmm for optimal BLAS performance
        if self.bias is not None:
            return torch.addmm(self.bias, input, self.weight.t())
        else:
            return torch.mm(input, self.weight.t())
    
    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, output_dim={self.output_dim}, bias={self.bias is not None}, precision={self.precision}'

# Performance benchmarking utilities
@torch.jit.script
def benchmark_${operatorLower}(model, input_tensor: torch.Tensor, num_iterations: int = 100) -> float:
    """Benchmark the ${operatorTitle} performance"""
    model.eval()
    
    # Warmup runs to stabilize GPU clocks
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Synchronize GPU before timing
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(input_tensor)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = elapsed_time_ms / num_iterations
    
    return avg_time_ms`;

            case 'JAX':
                return `import jax
import jax.numpy as jnp
from jax import jit, vmap, pmap
from jax.experimental import pjit
from jax.sharding import PartitionSpec as P
import functools
from typing import Optional

# ${submission.name}: High-performance ${operatorTitle} in JAX
# Performance: ${submission.performance} TFLOPS on ${submission.device}

@functools.partial(jit, static_argnums=(2, 3))
def ${submissionLower}(
    x: jnp.ndarray,
    weight: jnp.ndarray, 
    bias: Optional[jnp.ndarray] = None,
    precision: Optional[str] = '${submission.precision.toLowerCase()}'
) -> jnp.ndarray:
    """
    Optimized ${operatorTitle} with XLA compilation and precision control
    
    Args:
        x: Input tensor of shape (batch_size, input_dim)
        weight: Weight matrix of shape (output_dim, input_dim)  
        bias: Optional bias vector of shape (output_dim,)
        precision: Computation precision ('fp16', 'bf16', 'fp32')
    
    Returns:
        Output tensor of shape (batch_size, output_dim)
    """
    # Set precision for optimal performance on ${submission.device}
    if precision == 'fp16':
        preferred_dtype = jnp.float16
    elif precision == 'bf16':
        preferred_dtype = jnp.bfloat16
    else:
        preferred_dtype = jnp.float32
    
    # Cast inputs to preferred precision for Tensor Core utilization
    x = x.astype(preferred_dtype)
    weight = weight.astype(preferred_dtype)
    
    # Optimized matrix multiplication using einsum with highest precision
    # XLA will automatically select the best GEMM implementation
    output = jnp.einsum('bi,oi->bo', x, weight, precision=jax.lax.Precision.HIGHEST)
    
    # Add bias if provided
    if bias is not None:
        bias = bias.astype(preferred_dtype)
        output = output + bias
    
    return output

@functools.partial(jit, static_argnums=(3,))
def ${submissionLower}_fused(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    activation: str = 'linear'
) -> jnp.ndarray:
    """
    Fused ${operatorTitle} with activation for optimal memory bandwidth
    """
    output = ${submissionLower}(x, weight, bias, '${submission.precision.toLowerCase()}')
    
    # Fused activation functions - compiled into single kernel by XLA
    if activation == 'relu':
        return jax.nn.relu(output)
    elif activation == 'gelu':
        return jax.nn.gelu(output)
    elif activation == 'swish':
        return jax.nn.swish(output)
    elif activation == 'silu':
        return jax.nn.silu(output)
    elif activation == 'leaky_relu':
        return jax.nn.leaky_relu(output, negative_slope=0.01)
    else:
        return output

# Vectorized version for efficient batch processing
${submissionLower}_vmap = vmap(${submissionLower}, in_axes=(0, None, None, None), out_axes=0)

# Parallel version for multi-device training (TPU/multi-GPU)
@functools.partial(pmap, axis_name='devices', static_broadcasted_argnums=(3,))
def ${submissionLower}_pmap(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None,
    precision: str = '${submission.precision.toLowerCase()}'
) -> jnp.ndarray:
    """
    Parallel ${operatorTitle} across multiple devices with gradient synchronization
    """
    return ${submissionLower}(x, weight, bias, precision)

# Sharded version for large model parallelism
@functools.partial(
    pjit,
    in_shardings=(P('data', None), P(None, 'model'), P('model',)),
    out_shardings=P('data', 'model')
)
def ${submissionLower}_sharded(
    x: jnp.ndarray,
    weight: jnp.ndarray,
    bias: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Sharded ${operatorTitle} for large-scale model parallelism
    Automatically partitions computation across devices
    """
    return ${submissionLower}(x, weight, bias, '${submission.precision.toLowerCase()}')

def benchmark_${submissionLower}(input_shape, output_dim, num_iterations=1000):
    """
    Comprehensive benchmarking for ${operatorTitle} performance
    Measures throughput and efficiency on ${submission.device}
    """
    key = jax.random.PRNGKey(42)
    x_key, w_key, b_key = jax.random.split(key, 3)
    
    # Generate random inputs with realistic distributions
    x = jax.random.normal(x_key, input_shape, dtype=jnp.float32)
    weight = jax.random.normal(w_key, (output_dim, input_shape[-1]), dtype=jnp.float32)
    bias = jax.random.normal(b_key, (output_dim,), dtype=jnp.float32)
    
    # Compile the function with XLA optimizations
    compiled_fn = jit(${submissionLower})
    
    # Warmup to ensure compilation and GPU initialization
    for _ in range(10):
        _ = compiled_fn(x, weight, bias, '${submission.precision.toLowerCase()}').block_until_ready()
    
    # Accurate timing with JAX profiling
    import time
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        result = compiled_fn(x, weight, bias, '${submission.precision.toLowerCase()}')
        result.block_until_ready()  # Ensure computation completes
    end_time = time.perf_counter()
    
    avg_time_ms = (end_time - start_time) * 1000 / num_iterations
    return avg_time_ms, result.shape`;

            case 'CUTLASS':
                return `#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>

// ${submission.name}: CUTLASS-based optimized ${operatorTitle}
// Performance: ${submission.performance} TFLOPS on ${submission.device}
// Utilizes Tensor Cores with ${submission.precision} precision

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

// CUTLASS GEMM configuration optimized for ${submission.device}
using ${submissionLower}_gemm = cutlass::gemm::device::Gemm<
    ElementA,                           // Data type of A matrix
    cutlass::layout::RowMajor,          // Layout of A matrix
    ElementB,                           // Data type of B matrix  
    cutlass::layout::ColumnMajor,       // Layout of B matrix
    ElementC,                           // Data type of C matrix
    cutlass::layout::RowMajor,          // Layout of C matrix
    ElementAccumulator,                 // Data type of accumulator
    cutlass::arch::OpClassTensorOp,     // Use Tensor Cores
    cutlass::arch::Sm80,                // Target architecture (${submission.device})
    cutlass::gemm::GemmShape<128, 256, 64>,    // Threadblock tile size
    cutlass::gemm::GemmShape<64, 64, 64>,      // Warp tile size  
    cutlass::gemm::GemmShape<16, 8, 16>,       // Instruction tile size
    cutlass::epilogue::thread::LinearCombination<
        ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
        ElementAccumulator, ElementAccumulator>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    3,                                  // Number of pipeline stages
    16,                                 // Minimum alignment for A
    16,                                 // Minimum alignment for B
    false,                              // SplitKSerial
    cutlass::arch::OpMultiplyAdd        // Math operation
>;

class ${submission.name.replace(/[^a-zA-Z0-9]/g, '')}Kernel {
private:
    ${submissionLower}_gemm gemm_op;
    
public:
    ${submission.name.replace(/[^a-zA-Z0-9]/g, '')}Kernel() = default;
    
    /**
     * Execute optimized ${operatorTitle} using CUTLASS
     * Achieves ${submission.performance} TFLOPS on ${submission.device}
     */
    cutlass::Status operator()(
        int M, int N, int K,
        ElementA const* A, int lda,
        ElementB const* B, int ldb, 
        ElementC* C, int ldc,
        ElementAccumulator alpha = ElementAccumulator(1),
        ElementAccumulator beta = ElementAccumulator(0),
        cudaStream_t stream = nullptr
    ) {
        // Configure GEMM arguments
        typename ${submissionLower}_gemm::Arguments args{
            {M, N, K},              // Problem size
            {A, lda},               // Tensor A and leading dimension
            {B, ldb},               // Tensor B and leading dimension  
            {C, ldc},               // Tensor C and leading dimension
            {C, ldc},               // Tensor D and leading dimension (output)
            {alpha, beta}           // Scalars for linear combination
        };
        
        // Initialize GEMM workspace
        size_t workspace_size = ${submissionLower}_gemm::get_workspace_size(args);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        
        // Initialize the GEMM operator
        cutlass::Status status = gemm_op.initialize(args, workspace.get(), stream);
        if (status != cutlass::Status::kSuccess) {
            return status;
        }
        
        // Execute the GEMM operation
        status = gemm_op(stream);
        if (status != cutlass::Status::kSuccess) {
            return status;
        }
        
        return cutlass::Status::kSuccess;
    }
    
    /**
     * Benchmark the kernel performance
     */
    float benchmark(int M, int N, int K, int iterations = 100) {
        // Allocate test matrices
        cutlass::HostTensor<ElementA, cutlass::layout::RowMajor> tensor_A({M, K});
        cutlass::HostTensor<ElementB, cutlass::layout::ColumnMajor> tensor_B({K, N});
        cutlass::HostTensor<ElementC, cutlass::layout::RowMajor> tensor_C({M, N});
        
        // Initialize with random data
        tensor_A.sync_device();
        tensor_B.sync_device();
        tensor_C.sync_device();
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            this->operator()(M, N, K, 
                tensor_A.device_data(), tensor_A.device_ref().stride(0),
                tensor_B.device_data(), tensor_B.device_ref().stride(0),
                tensor_C.device_data(), tensor_C.device_ref().stride(0));
        }
        
        // Timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < iterations; ++i) {
            this->operator()(M, N, K,
                tensor_A.device_data(), tensor_A.device_ref().stride(0),
                tensor_B.device_data(), tensor_B.device_ref().stride(0), 
                tensor_C.device_data(), tensor_C.device_ref().stride(0));
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return elapsed_time / iterations;  // Average time per iteration in ms
    }
};`;

            default:
                return `// ${submission.dsl} implementation for ${submission.name}
// Optimized ${operatorTitle} kernel achieving ${submission.performance} TFLOPS
// Target device: ${submission.device}
// Precision: ${submission.precision}

/*
 * High-performance ${operatorTitle.toLowerCase()} implementation
 * 
 * Key optimizations:
 * - Memory coalescing for optimal bandwidth utilization (${submission.memoryBW} TB/s)
 * - Compute efficiency: ${submission.efficiency}%
 * - Low latency: ${submission.latency} ms
 * - Batch processing with size ${submission.batchSize}
 * 
 * This implementation leverages ${submission.dsl} framework features
 * to achieve optimal performance on ${submission.device} architecture.
 */

function ${submissionLower}_optimized() {
    // Framework-specific optimizations
    // - Vectorized operations for SIMD utilization
    // - Memory prefetching and caching strategies  
    // - Parallel execution across compute units
    // - Precision-specific optimizations for ${submission.precision}
    
    return {
        performance: "${submission.performance} TFLOPS",
        correctness: "${submission.correctness}%",
        efficiency: "${submission.efficiency}%",
        device: "${submission.device}",
        framework: "${submission.dsl}"
    };
}`;
        }
    };

    if (!submission) {
        return (
            <div className="container" style={{ padding: '3rem 0' }}>
                <h2>Submission not found</h2>
                <Link to="/">Back to home</Link>
            </div>
        );
    }

    const operatorTitle = operatorInfo[operatorKey]?.title || 'Unknown Operator';

    return (
        <div className="submission-detail">
            {/* Header */}
            <section className="submission-header">
                <div className="container">
                    <div className="submission-nav">
                        <button 
                            onClick={() => navigate(-1)} 
                            className="back-btn"
                        >
                            <i className="fas fa-arrow-left"></i> Back to Leaderboard
                        </button>
                        <div className="breadcrumb">
                            <Link to="/">Home</Link>
                            <span className="separator">/</span>
                            <Link to={`/operator/${operatorKey}`}>{operatorTitle}</Link>
                            <span className="separator">/</span>
                            <span>Submission #{submission.id}</span>
                        </div>
                    </div>

                    <div className="submission-info">
                        <div className="submission-title">
                            <h1>{submission.name}</h1>
                            <div className="submission-badges">
                                <span className={`dsl-badge ${submission.dsl.toLowerCase()}`}>
                                    {submission.dsl}
                                </span>
                                <span className={`device-badge ${submission.device.toLowerCase().replace(' ', '')}`}>
                                    {submission.device}
                                </span>
                                <span className="precision-badge">
                                    {submission.precision}
                                </span>
                            </div>
                        </div>

                        <div className="submission-stats-row">
                            <div className="stat-square performance">
                                <div className="stat-icon">
                                    <i className="fas fa-tachometer-alt"></i>
                                </div>
                                <div className="stat-value">{submission.performance.toLocaleString()}</div>
                                <div className="stat-label">TFLOPS</div>
                            </div>
                            <div className="stat-square correctness">
                                <div className="stat-icon">
                                    <i className="fas fa-check-circle"></i>
                                </div>
                                <div className="stat-value">{submission.correctness}%</div>
                                <div className="stat-label">Correctness</div>
                            </div>
                            <div className="stat-square efficiency">
                                <div className="stat-icon">
                                    <i className="fas fa-chart-line"></i>
                                </div>
                                <div className="stat-value">{submission.efficiency}%</div>
                                <div className="stat-label">Efficiency</div>
                            </div>
                            <div className="stat-square latency">
                                <div className="stat-icon">
                                    <i className="fas fa-clock"></i>
                                </div>
                                <div className="stat-value">{submission.latency}</div>
                                <div className="stat-label">ms</div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Kernel Code Section - Displayed directly */}
            <section className="submission-content">
                <div className="container">
                    <div className="code-section">
                        <div className="code-header">
                            <h2>Kernel Implementation</h2>
                            <div className="code-actions">
                                <button className="action-btn copy-btn" title="Copy Code">
                                    <i className="fas fa-copy"></i>
                                </button>
                                <button className="action-btn download-btn" title="Download">
                                    <i className="fas fa-download"></i>
                                </button>
                            </div>
                        </div>
                        <div className="code-container">
                            <pre><code className={`language-${submission.dsl.toLowerCase()}`}>
{generateKernelCode(submission, operatorTitle)}
                            </code></pre>
                        </div>
                    </div>
                </div>
            </section>

            {/* Details and Metrics Section */}
            <section className="submission-details">
                <div className="container">
                    <div className="details-grid">
                        <div className="detail-card author-info">
                            <h3><i className="fas fa-user"></i> Author Information</h3>
                            <div className="author-profile">
                                <img 
                                    src="https://via.placeholder.com/80" 
                                    alt={submission.author} 
                                    className="author-avatar-large"
                                />
                                <div className="author-details">
                                    <h4>@{submission.author}</h4>
                                    <p className="author-title">
                                        {submission.author.includes('pro') || submission.author.includes('expert') ? 
                                            'Senior GPU Performance Engineer' : 
                                            submission.author.includes('ninja') || submission.author.includes('master') ?
                                            'Principal Software Engineer' :
                                            'Software Engineer'
                                        }
                                    </p>
                                    <div className="author-stats">
                                        <span className="stat">
                                            <i className="fas fa-trophy"></i>
                                            {Math.floor(Math.random() * 15) + 3} submissions
                                        </span>
                                        <span className="stat">
                                            <i className="fas fa-star"></i>
                                            {Math.floor(Math.random() * 50) + 20} points
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div className="detail-card submission-meta">
                            <h3><i className="fas fa-info-circle"></i> Submission Details</h3>
                            <div className="meta-grid">
                                <div className="meta-item">
                                    <span className="meta-label">Submission ID</span>
                                    <span className="meta-value">#{submission.id}</span>
                                </div>
                                <div className="meta-item">
                                    <span className="meta-label">Operator</span>
                                    <span className="meta-value">{operatorTitle}</span>
                                </div>
                                <div className="meta-item">
                                    <span className="meta-label">Implementation</span>
                                    <span className="meta-value">{submission.dsl}</span>
                                </div>
                                <div className="meta-item">
                                    <span className="meta-label">Target Device</span>
                                    <span className="meta-value">{submission.device}</span>
                                </div>
                                <div className="meta-item">
                                    <span className="meta-label">Precision</span>
                                    <span className="meta-value">{submission.precision}</span>
                                </div>
                                <div className="meta-item">
                                    <span className="meta-label">Batch Size</span>
                                    <span className="meta-value">{submission.batchSize}</span>
                                </div>
                                <div className="meta-item">
                                    <span className="meta-label">Submitted</span>
                                    <span className="meta-value">{submission.date}</span>
                                </div>
                            </div>
                        </div>

                        <div className="detail-card kernel-info">
                            <h3><i className="fas fa-microchip"></i> Kernel Information</h3>
                            <div className="kernel-description">
                                <p>
                                    This {submission.dsl} implementation of {operatorTitle.toLowerCase()} 
                                    is optimized for {submission.device} architecture, achieving 
                                    {submission.performance.toLocaleString()} TFLOPS with {submission.correctness}% correctness.
                                </p>
                                <div className="optimization-highlights">
                                    <h4>Key Optimizations:</h4>
                                    <ul>
                                        {submission.dsl === 'Triton' && (
                                            <>
                                                <li>Block-level parallelization with optimal tile sizes</li>
                                                <li>Automatic memory coalescing and vectorization</li>
                                                <li>Hardware-specific instruction selection</li>
                                                <li>Auto-tuning for optimal block configurations</li>
                                            </>
                                        )}
                                        {submission.dsl === 'CUDA' && (
                                            <>
                                                <li>Tensor Core utilization for mixed precision</li>
                                                <li>Shared memory optimization and bank conflict avoidance</li>
                                                <li>Warp-level primitives for synchronization</li>
                                                <li>Memory coalescing patterns</li>
                                            </>
                                        )}
                                        {submission.dsl === 'CUTLASS' && (
                                            <>
                                                <li>Template-based kernel generation</li>
                                                <li>Hierarchical tiling strategies</li>
                                                <li>Automated pipeline optimization</li>
                                                <li>WMMA API integration</li>
                                            </>
                                        )}
                                        {submission.dsl === 'PyTorch' && (
                                            <>
                                                <li>JIT compilation with TorchScript</li>
                                                <li>Operator fusion for reduced memory access</li>
                                                <li>Custom autograd functions</li>
                                                <li>Automatic mixed precision support</li>
                                            </>
                                        )}
                                        {submission.dsl === 'JAX' && (
                                            <>
                                                <li>XLA compilation for optimal performance</li>
                                                <li>Automatic differentiation support</li>
                                                <li>Device sharding and parallelization</li>
                                                <li>Vectorized and parallel operations</li>
                                            </>
                                        )}
                                        <li>Memory bandwidth optimization: {submission.memoryBW} TB/s</li>
                                        <li>Compute efficiency: {submission.efficiency}%</li>
                                    </ul>
                                </div>
                            </div>
                        </div>

                        <div className="metric-card">
                            <h3>Performance Metrics</h3>
                            <div className="metric-chart">
                                <div className="metric-bar">
                                    <span className="metric-label">Performance</span>
                                    <div className="metric-progress">
                                        <div 
                                            className="metric-fill performance" 
                                            style={{ width: `${Math.min(submission.performance / 3000 * 100, 100)}%` }}
                                        ></div>
                                    </div>
                                    <span className="metric-value">{submission.performance} TFLOPS</span>
                                </div>
                                <div className="metric-bar">
                                    <span className="metric-label">Correctness</span>
                                    <div className="metric-progress">
                                        <div 
                                            className="metric-fill correctness" 
                                            style={{ width: `${submission.correctness}%` }}
                                        ></div>
                                    </div>
                                    <span className="metric-value">{submission.correctness}%</span>
                                </div>
                                <div className="metric-bar">
                                    <span className="metric-label">Efficiency</span>
                                    <div className="metric-progress">
                                        <div 
                                            className="metric-fill efficiency" 
                                            style={{ width: `${submission.efficiency}%` }}
                                        ></div>
                                    </div>
                                    <span className="metric-value">{submission.efficiency}%</span>
                                </div>
                            </div>
                        </div>

                        <div className="metric-card">
                            <h3>Resource Utilization</h3>
                            <div className="resource-metrics">
                                <div className="resource-item">
                                    <span className="resource-label">Memory Bandwidth</span>
                                    <span className="resource-value">{submission.memoryBW} TB/s</span>
                                </div>
                                <div className="resource-item">
                                    <span className="resource-label">Latency</span>
                                    <span className="resource-value">{submission.latency} ms</span>
                                </div>
                                <div className="resource-item">
                                    <span className="resource-label">Batch Size</span>
                                    <span className="resource-value">{submission.batchSize}</span>
                                </div>
                                <div className="resource-item">
                                    <span className="resource-label">Precision</span>
                                    <span className="resource-value">{submission.precision}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default SubmissionDetailPage;