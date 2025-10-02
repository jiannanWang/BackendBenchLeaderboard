// Mock data organized by operators
export const mockSubmissionsByOperator = {
    matmul: [
        {
            id: 1247,
            name: "OptimizedMatMul_v3",
            operator: "Matrix Multiplication",
            dsl: "Triton",
            device: "H100",
            performance: 2847.3,
            correctness: 100,
            author: "researcher_ai",
            date: "2024-03-15",
            latency: 0.032,
            memoryBW: 1.2,
            efficiency: 94.7,
            precision: "FP16",
            batchSize: 1024
        },
        {
            id: 1241,
            name: "MatMul_TensorCore",
            operator: "Matrix Multiplication",
            dsl: "CUTLASS",
            device: "A100",
            performance: 2734.5,
            correctness: 99.9,
            author: "cutlass_pro",
            date: "2024-03-13",
            latency: 0.035,
            memoryBW: 1.1,
            efficiency: 92.8,
            precision: "BF16",
            batchSize: 512
        },
        {
            id: 1238,
            name: "GEMM_Optimized",
            operator: "Matrix Multiplication",
            dsl: "CUDA",
            device: "H100",
            performance: 2689.7,
            correctness: 100,
            author: "cuda_expert",
            date: "2024-03-11",
            latency: 0.038,
            memoryBW: 1.0,
            efficiency: 90.3,
            precision: "FP16",
            batchSize: 1024
        },
        {
            id: 1234,
            name: "HighPerf_GEMM",
            operator: "Matrix Multiplication",
            dsl: "PyTorch",
            device: "V100",
            performance: 2456.2,
            correctness: 99.7,
            author: "pytorch_dev",
            date: "2024-03-09",
            latency: 0.045,
            memoryBW: 0.9,
            efficiency: 88.5,
            precision: "FP32",
            batchSize: 256
        },
        {
            id: 1230,
            name: "MatMul_JAX",
            operator: "Matrix Multiplication",
            dsl: "JAX",
            device: "RTX 4090",
            performance: 2398.9,
            correctness: 99.8,
            author: "jax_master",
            date: "2024-03-07",
            latency: 0.041,
            memoryBW: 1.0,
            efficiency: 91.2,
            precision: "BF16",
            batchSize: 1024
        }
    ],
    attention: [
        {
            id: 1244,
            name: "FlashAttention_Custom",
            operator: "Attention", 
            dsl: "CUDA",
            device: "A100",
            performance: 2743.8,
            correctness: 99.8,
            author: "gpu_wizard",
            date: "2024-03-12",
            latency: 0.038,
            memoryBW: 1.1,
            efficiency: 92.3,
            precision: "BF16",
            batchSize: 512
        },
        {
            id: 1240,
            name: "MultiHead_Attention",
            operator: "Attention",
            dsl: "Triton",
            device: "H100",
            performance: 2678.4,
            correctness: 99.9,
            author: "attention_master",
            date: "2024-03-10",
            latency: 0.041,
            memoryBW: 1.2,
            efficiency: 94.1,
            precision: "FP16",
            batchSize: 1024
        },
        {
            id: 1236,
            name: "SelfAttention_Fast",
            operator: "Attention",
            dsl: "PyTorch",
            device: "RTX 4090",
            performance: 2534.7,
            correctness: 99.6,
            author: "attention_ninja",
            date: "2024-03-08",
            latency: 0.048,
            memoryBW: 0.9,
            efficiency: 87.8,
            precision: "FP16",
            batchSize: 256
        },
        {
            id: 1232,
            name: "Attention_Optimized",
            operator: "Attention",
            dsl: "CUTLASS",
            device: "V100",
            performance: 2387.1,
            correctness: 99.9,
            author: "attention_pro",
            date: "2024-03-06",
            latency: 0.052,
            memoryBW: 0.8,
            efficiency: 85.4,
            precision: "FP32",
            batchSize: 512
        }
    ],
    layernorm: [
        {
            id: 1239,
            name: "LayerNorm_Fused",
            operator: "Layer Normalization",
            dsl: "Triton", 
            device: "H100",
            performance: 2689.4,
            correctness: 100,
            author: "kernel_master",
            date: "2024-03-10",
            latency: 0.025,
            memoryBW: 1.3,
            efficiency: 96.1,
            precision: "FP16",
            batchSize: 2048
        },
        {
            id: 1237,
            name: "LayerNorm_Fast",
            operator: "Layer Normalization",
            dsl: "CUDA",
            device: "A100",
            performance: 2534.8,
            correctness: 99.8,
            author: "norm_ninja",
            date: "2024-03-09",
            latency: 0.028,
            memoryBW: 1.1,
            efficiency: 93.4,
            precision: "BF16",
            batchSize: 1024
        },
        {
            id: 1233,
            name: "LayerNorm_Apex",
            operator: "Layer Normalization",
            dsl: "PyTorch",
            device: "RTX 4090",
            performance: 2445.6,
            correctness: 99.9,
            author: "apex_user",
            date: "2024-03-07",
            latency: 0.032,
            memoryBW: 1.0,
            efficiency: 90.7,
            precision: "FP16",
            batchSize: 512
        }
    ],
    conv2d: [
        {
            id: 1235,
            name: "Conv2D_Optimized",
            operator: "2D Convolution",
            dsl: "CUTLASS",
            device: "RTX 4090",
            performance: 2621.7,
            correctness: 99.9,
            author: "perf_hacker",
            date: "2024-03-08",
            latency: 0.045,
            memoryBW: 0.9,
            efficiency: 89.4,
            precision: "FP32",
            batchSize: 256
        },
        {
            id: 1231,
            name: "Conv_TensorRT",
            operator: "2D Convolution",
            dsl: "CUDA",
            device: "A100",
            performance: 2487.3,
            correctness: 100,
            author: "conv_expert",
            date: "2024-03-07",
            latency: 0.052,
            memoryBW: 0.8,
            efficiency: 87.2,
            precision: "FP16",
            batchSize: 512
        },
        {
            id: 1229,
            name: "Conv2D_Winograd",
            operator: "2D Convolution",
            dsl: "Triton",
            device: "H100",
            performance: 2398.1,
            correctness: 99.7,
            author: "winograd_pro",
            date: "2024-03-05",
            latency: 0.048,
            memoryBW: 1.0,
            efficiency: 91.3,
            precision: "BF16",
            batchSize: 1024
        }
    ],
    gelu: [
        {
            id: 1227,
            name: "GELU_Fast",
            operator: "GELU",
            dsl: "PyTorch",
            device: "V100",
            performance: 2534.2,
            correctness: 100,
            author: "ml_optimizer",
            date: "2024-03-05",
            latency: 0.021,
            memoryBW: 0.8,
            efficiency: 91.7,
            precision: "FP16",
            batchSize: 1024
        },
        {
            id: 1225,
            name: "GELU_Triton",
            operator: "GELU",
            dsl: "Triton",
            device: "H100",
            performance: 2467.9,
            correctness: 99.9,
            author: "activation_pro",
            date: "2024-03-04",
            latency: 0.019,
            memoryBW: 0.9,
            efficiency: 93.8,
            precision: "FP16",
            batchSize: 2048
        },
        {
            id: 1223,
            name: "GELU_CUDA",
            operator: "GELU",
            dsl: "CUDA",
            device: "A100",
            performance: 2398.5,
            correctness: 99.8,
            author: "cuda_ninja",
            date: "2024-03-03",
            latency: 0.022,
            memoryBW: 0.9,
            efficiency: 89.6,
            precision: "BF16",
            batchSize: 512
        }
    ],
    softmax: [
        {
            id: 1224,
            name: "Softmax_Optimized",
            operator: "Softmax",
            dsl: "JAX",
            device: "A100",
            performance: 2445.6,
            correctness: 99.7,
            author: "jax_expert",
            date: "2024-03-03",
            latency: 0.029,
            memoryBW: 1.0,
            efficiency: 88.2,
            precision: "BF16",
            batchSize: 512
        },
        {
            id: 1222,
            name: "Softmax_Stable",
            operator: "Softmax",
            dsl: "CUDA",
            device: "H100",
            performance: 2398.1,
            correctness: 100,
            author: "softmax_guru",
            date: "2024-03-02",
            latency: 0.026,
            memoryBW: 1.1,
            efficiency: 90.5,
            precision: "FP16",
            batchSize: 1024
        },
        {
            id: 1220,
            name: "Softmax_Fast",
            operator: "Softmax",
            dsl: "Triton",
            device: "RTX 4090",
            performance: 2298.7,
            correctness: 99.9,
            author: "triton_master",
            date: "2024-03-01",
            latency: 0.031,
            memoryBW: 0.8,
            efficiency: 86.3,
            precision: "FP32",
            batchSize: 256
        }
    ],
    linear: [
        {
            id: 1221,
            name: "Linear_TensorCore",
            operator: "Linear",
            dsl: "CUTLASS",
            device: "H100",
            performance: 2387.9,
            correctness: 100,
            author: "tensor_pro",
            date: "2024-03-01",
            latency: 0.035,
            memoryBW: 1.1,
            efficiency: 93.5,
            precision: "FP16",
            batchSize: 1024
        },
        {
            id: 1219,
            name: "Linear_Fast",
            operator: "Linear",
            dsl: "PyTorch",
            device: "A100",
            performance: 2298.7,
            correctness: 99.8,
            author: "linear_master",
            date: "2024-02-29",
            latency: 0.042,
            memoryBW: 0.9,
            efficiency: 89.1,
            precision: "BF16",
            batchSize: 512
        },
        {
            id: 1217,
            name: "Linear_Optimized",
            operator: "Linear",
            dsl: "CUDA",
            device: "V100",
            performance: 2198.4,
            correctness: 99.9,
            author: "linear_ninja",
            date: "2024-02-28",
            latency: 0.048,
            memoryBW: 0.8,
            efficiency: 87.2,
            precision: "FP32",
            batchSize: 256
        }
    ],
    embedding: [
        {
            id: 1218,
            name: "Embedding_Fast",
            operator: "Embedding",
            dsl: "TVM",
            device: "RTX 3080",
            performance: 2298.3,
            correctness: 99.9,
            author: "tvm_ninja",
            date: "2024-02-28",
            latency: 0.042,
            memoryBW: 0.7,
            efficiency: 85.6,
            precision: "FP32",
            batchSize: 128
        },
        {
            id: 1216,
            name: "Embedding_Lookup",
            operator: "Embedding",
            dsl: "CUDA",
            device: "V100",
            performance: 2134.5,
            correctness: 100,
            author: "embed_pro",
            date: "2024-02-27",
            latency: 0.048,
            memoryBW: 0.6,
            efficiency: 82.3,
            precision: "FP16",
            batchSize: 256
        },
        {
            id: 1214,
            name: "Embedding_PyTorch",
            operator: "Embedding",
            dsl: "PyTorch",
            device: "A100",
            performance: 2087.9,
            correctness: 99.8,
            author: "pytorch_expert",
            date: "2024-02-26",
            latency: 0.052,
            memoryBW: 0.7,
            efficiency: 80.7,
            precision: "BF16",
            batchSize: 512
        }
    ]
};

// Operator metadata
export const operatorInfo = {
    matmul: {
        title: "Matrix Multiplication",
        description: "High-performance GEMM implementations for dense linear algebra operations",
        icon: "fas fa-calculator"
    },
    attention: {
        title: "Attention",
        description: "Self-attention and multi-head attention mechanisms for transformer models",
        icon: "fas fa-eye"
    },
    layernorm: {
        title: "Layer Normalization",
        description: "Fused layer normalization implementations for stable training",
        icon: "fas fa-layer-group"
    },
    conv2d: {
        title: "2D Convolution",
        description: "Optimized convolution operations for computer vision workloads",
        icon: "fas fa-grip"
    },
    gelu: {
        title: "GELU Activation",
        description: "Gaussian Error Linear Unit activation function implementations",
        icon: "fas fa-wave-square"
    },
    softmax: {
        title: "Softmax",
        description: "Numerically stable softmax implementations for probability distributions",
        icon: "fas fa-chart-line"
    },
    linear: {
        title: "Linear Layer",
        description: "Fully connected layer implementations with various optimizations",
        icon: "fas fa-minus"
    },
    embedding: {
        title: "Embedding",
        description: "Token and positional embedding lookup operations for NLP models",
        icon: "fas fa-map"
    }
};

// Calculate stats for each operator
export const getOperatorStats = () => {
    const stats = {};
    
    Object.keys(mockSubmissionsByOperator).forEach(operatorKey => {
        const submissions = mockSubmissionsByOperator[operatorKey];
        const topPerformance = Math.max(...submissions.map(s => s.performance));
        
        stats[operatorKey] = {
            submissionCount: submissions.length,
            topPerformance: topPerformance
        };
    });
    
    return stats;
};

// Get global stats
export const getGlobalStats = () => {
    let totalSubmissions = 0;
    const uniqueContributors = new Set();
    const totalOperators = Object.keys(mockSubmissionsByOperator).length;
    
    Object.values(mockSubmissionsByOperator).forEach(submissions => {
        totalSubmissions += submissions.length;
        submissions.forEach(submission => {
            uniqueContributors.add(submission.author);
        });
    });
    
    return {
        totalSubmissions,
        uniqueContributors: uniqueContributors.size,
        totalOperators
    };
};