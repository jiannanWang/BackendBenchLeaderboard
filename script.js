// Mock data organized by operators
const mockSubmissionsByOperator = {
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
            id: 1233,
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
        }
    ],
    gelu: [
        {
            id: 1231,
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
            id: 1229,
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
        }
    ],
    softmax: [
        {
            id: 1228,
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
            id: 1226,
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
        }
    ],
    linear: [
        {
            id: 1225,
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
            id: 1223,
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
        }
    ],
    embedding: [
        {
            id: 1222,
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
            id: 1220,
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
        }
    ]
};

// Operator metadata
const operatorInfo = {
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

let currentOperator = null;
let currentSubmissions = [];
let currentView = 'table';

// DOM Elements
const dslFilter = document.getElementById('dsl-filter');
const deviceFilter = document.getElementById('device-filter');
const sortFilter = document.getElementById('sort-filter');
const tableView = document.getElementById('table-view');
const cardsView = document.getElementById('cards-view');
const viewBtns = document.querySelectorAll('.view-btn');
const modal = document.getElementById('submission-modal');
const modalClose = document.querySelector('.modal-close');
const submitBtn = document.querySelector('.submit-btn');
const operatorsSection = document.getElementById('operators');
const leaderboardSection = document.getElementById('leaderboard');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    showOperatorsView();
    updateGlobalStats();
});

// Event Listeners
function initializeEventListeners() {
    // Filter event listeners (only for individual operator leaderboards)
    if (dslFilter) dslFilter.addEventListener('change', applyFilters);
    if (deviceFilter) deviceFilter.addEventListener('change', applyFilters);
    if (sortFilter) sortFilter.addEventListener('change', applySorting);
    
    // View toggle event listeners
    viewBtns.forEach(btn => {
        btn.addEventListener('click', () => toggleView(btn.dataset.view));
    });
    
    // Modal event listeners
    if (modalClose) modalClose.addEventListener('click', closeModal);
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal();
        });
    }
    
    // Submit button
    if (submitBtn) {
        submitBtn.addEventListener('click', () => {
            window.location.href = 'submit.html';
        });
    }
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (modal && modal.classList.contains('show')) {
                closeModal();
            } else if (currentOperator) {
                backToOperators();
            }
        }
    });
}

// Navigation Functions
function viewOperatorLeaderboard(operatorKey) {
    if (!mockSubmissionsByOperator[operatorKey]) {
        console.error(`No data found for operator: ${operatorKey}`);
        return;
    }
    
    currentOperator = operatorKey;
    currentSubmissions = [...mockSubmissionsByOperator[operatorKey]];
    
    // Update page title and description
    const info = operatorInfo[operatorKey];
    document.getElementById('current-operator-title').textContent = `${info.title} Leaderboard`;
    document.getElementById('current-operator-description').textContent = `Performance rankings for ${info.title.toLowerCase()} kernels`;
    
    // Sort by performance initially
    applySorting();
    
    // Show leaderboard section, hide operators section
    operatorsSection.classList.add('hidden');
    leaderboardSection.classList.remove('hidden');
    
    // Reset filters
    if (dslFilter) dslFilter.value = '';
    if (deviceFilter) deviceFilter.value = '';
    if (sortFilter) sortFilter.value = 'performance';
    
    // Reset view to table
    currentView = 'table';
    viewBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === 'table');
    });
    tableView.style.display = 'block';
    cardsView.style.display = 'none';
}

function backToOperators() {
    currentOperator = null;
    currentSubmissions = [];
    
    // Show operators section, hide leaderboard section
    operatorsSection.classList.remove('hidden');
    leaderboardSection.classList.add('hidden');
    
    // Update global stats
    updateGlobalStats();
}

function showOperatorsView() {
    operatorsSection.classList.remove('hidden');
    leaderboardSection.classList.add('hidden');
}

// Filter and Sort Functions (for individual operator leaderboards)
function applyFilters() {
    if (!currentOperator) return;
    
    const dslValue = dslFilter ? dslFilter.value.toLowerCase() : '';
    const deviceValue = deviceFilter ? deviceFilter.value.toLowerCase() : '';
    
    const originalSubmissions = mockSubmissionsByOperator[currentOperator];
    currentSubmissions = originalSubmissions.filter(submission => {
        const dslMatch = !dslValue || submission.dsl.toLowerCase() === dslValue;
        const deviceMatch = !deviceValue || submission.device.toLowerCase().replace(' ', '') === deviceValue;
        
        return dslMatch && deviceMatch;
    });
    
    applySorting();
}

function applySorting() {
    if (!currentSubmissions.length) return;
    
    const sortValue = sortFilter ? sortFilter.value : 'performance';
    
    currentSubmissions.sort((a, b) => {
        switch (sortValue) {
            case 'performance':
                return b.performance - a.performance;
            case 'date':
                return new Date(b.date) - new Date(a.date);
            case 'correctness':
                return b.correctness - a.correctness;
            case 'author':
                return a.author.localeCompare(b.author);
            default:
                return b.performance - a.performance;
        }
    });
    
    renderLeaderboard();
}

// View Toggle Functions
function toggleView(view) {
    currentView = view;
    
    viewBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
    });
    
    if (view === 'table') {
        tableView.style.display = 'block';
        cardsView.style.display = 'none';
    } else {
        tableView.style.display = 'none';
        cardsView.style.display = 'grid';
        renderCardView();
    }
}

// Render Functions
function renderLeaderboard() {
    if (currentView === 'table') {
        renderTableView();
    } else {
        renderCardView();
    }
}

function renderTableView() {
    const tbody = document.querySelector('.leaderboard-table tbody');
    tbody.innerHTML = '';
    
    currentSubmissions.forEach((submission, index) => {
        const row = createTableRow(submission, index + 1);
        tbody.appendChild(row);
    });
}

function createTableRow(submission, rank) {
    const row = document.createElement('tr');
    row.className = `rank-${rank <= 3 ? rank : ''}`;
    
    row.innerHTML = `
        <td>
            <div class="rank">
                ${rank <= 3 ? '<i class="fas fa-crown"></i>' : ''}
                <span>${rank}</span>
            </div>
        </td>
        <td>
            <div class="submission-info">
                <span class="submission-name">${submission.name}</span>
                <span class="submission-id">#${submission.id}</span>
            </div>
        </td>
        <td>
            <span class="dsl-badge ${submission.dsl.toLowerCase()}">${submission.dsl}</span>
        </td>
        <td>
            <span class="device-badge ${submission.device.toLowerCase().replace(' ', '')}">${submission.device}</span>
        </td>
        <td>
            <div class="performance">
                <span class="value">${submission.performance.toLocaleString()}</span>
                <span class="unit">TFLOPS</span>
            </div>
        </td>
        <td>
            <div class="correctness">
                <span class="score">${submission.correctness}%</span>
                <i class="fas fa-check-circle"></i>
            </div>
        </td>
        <td>
            <div class="author">
                <img src="https://via.placeholder.com/32" alt="Author" class="author-avatar">
                <span>@${submission.author}</span>
            </div>
        </td>
        <td>
            <span class="date">${submission.date}</span>
        </td>
        <td>
            <div class="actions">
                <button class="action-btn view-btn" title="View Details" onclick="showSubmissionDetails(${submission.id})">
                    <i class="fas fa-eye"></i>
                </button>
                <button class="action-btn download-btn" title="Download" onclick="downloadSubmission(${submission.id})">
                    <i class="fas fa-download"></i>
                </button>
            </div>
        </td>
    `;
    
    return row;
}

function renderCardView() {
    const container = document.querySelector('.cards-container');
    container.innerHTML = '';
    
    currentSubmissions.forEach((submission, index) => {
        const card = createSubmissionCard(submission, index + 1);
        container.appendChild(card);
    });
}

function createSubmissionCard(submission, rank) {
    const card = document.createElement('div');
    card.className = `submission-card ${rank <= 3 ? `rank-${rank}` : ''}`;
    
    card.innerHTML = `
        <div class="card-header">
            <div class="rank-badge">
                ${rank === 1 ? '<i class="fas fa-crown"></i>' : ''}
                <span>${rank}</span>
            </div>
            <div class="submission-meta">
                <h4>${submission.name}</h4>
                <span class="submission-id">#${submission.id}</span>
            </div>
            <div class="performance-big">
                <span class="value">${submission.performance.toLocaleString()}</span>
                <span class="unit">TFLOPS</span>
            </div>
        </div>
        <div class="card-body">
            <div class="tags">
                <span class="operator-tag">${submission.operator}</span>
                <span class="dsl-badge ${submission.dsl.toLowerCase()}">${submission.dsl}</span>
                <span class="device-badge ${submission.device.toLowerCase().replace(' ', '')}">${submission.device}</span>
            </div>
            <div class="correctness-bar">
                <div class="correctness-label">Correctness: ${submission.correctness}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${submission.correctness}%"></div>
                </div>
            </div>
            <div class="author-date">
                <div class="author">
                    <img src="https://via.placeholder.com/24" alt="Author" class="author-avatar">
                    <span>@${submission.author}</span>
                </div>
                <span class="date">${submission.date}</span>
            </div>
        </div>
        <div class="card-actions">
            <button class="action-btn view-btn" onclick="showSubmissionDetails(${submission.id})">
                <i class="fas fa-eye"></i> View Details
            </button>
            <button class="action-btn download-btn" onclick="downloadSubmission(${submission.id})">
                <i class="fas fa-download"></i> Download
            </button>
        </div>
    `;
    
    return card;
}

// Modal Functions
function showSubmissionDetails(submissionId) {
    // Find submission across all operators
    let submission = null;
    for (const operatorKey in mockSubmissionsByOperator) {
        submission = mockSubmissionsByOperator[operatorKey].find(s => s.id === submissionId);
        if (submission) break;
    }
    
    if (!submission) return;
    
    // Update modal content
    document.querySelector('.modal-header h3').textContent = submission.name;
    
    // Update metrics
    const metricsGrid = document.querySelector('.metrics-grid');
    metricsGrid.innerHTML = `
        <div class="metric">
            <span class="metric-label">TFLOPS</span>
            <span class="metric-value">${submission.performance.toLocaleString()}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Latency</span>
            <span class="metric-value">${submission.latency}ms</span>
        </div>
        <div class="metric">
            <span class="metric-label">Memory BW</span>
            <span class="metric-value">${submission.memoryBW}TB/s</span>
        </div>
        <div class="metric">
            <span class="metric-label">Efficiency</span>
            <span class="metric-value">${submission.efficiency}%</span>
        </div>
    `;
    
    // Update configuration
    const configTable = document.querySelector('.config-table');
    configTable.innerHTML = `
        <div class="config-row">
            <span class="config-key">Operator</span>
            <span class="config-value">${submission.operator}</span>
        </div>
        <div class="config-row">
            <span class="config-key">DSL</span>
            <span class="config-value">${submission.dsl}</span>
        </div>
        <div class="config-row">
            <span class="config-key">Device</span>
            <span class="config-value">NVIDIA ${submission.device}</span>
        </div>
        <div class="config-row">
            <span class="config-key">Precision</span>
            <span class="config-value">${submission.precision}</span>
        </div>
        <div class="config-row">
            <span class="config-key">Batch Size</span>
            <span class="config-value">${submission.batchSize}</span>
        </div>
    `;
    
    // Update submission meta
    const submissionMeta = document.querySelector('.submission-meta-full');
    submissionMeta.innerHTML = `
        <div class="meta-item">
            <span class="meta-label">Author</span>
            <span class="meta-value">@${submission.author}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Submitted</span>
            <span class="meta-value">${new Date(submission.date).toLocaleDateString('en-US', { 
                year: 'numeric', 
                month: 'long', 
                day: 'numeric' 
            })}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Submission ID</span>
            <span class="meta-value">#${submission.id}</span>
        </div>
    `;
    
    modal.classList.add('show');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    modal.classList.remove('show');
    document.body.style.overflow = 'auto';
}

function downloadSubmission(submissionId) {
    // Find submission across all operators
    let submission = null;
    for (const operatorKey in mockSubmissionsByOperator) {
        submission = mockSubmissionsByOperator[operatorKey].find(s => s.id === submissionId);
        if (submission) break;
    }
    
    if (!submission) return;
    
    // Simulate download
    alert(`Downloading ${submission.name} kernel implementation...`);
    
    // In a real implementation, this would trigger an actual download
    // const blob = new Blob([kernelCode], { type: 'text/plain' });
    // const url = URL.createObjectURL(blob);
    // const a = document.createElement('a');
    // a.href = url;
    // a.download = `${submission.name}.cu`;
    // a.click();
    // URL.revokeObjectURL(url);
}

// Stats Update Functions
function updateGlobalStats() {
    // Calculate global statistics across all operators
    let totalSubmissions = 0;
    const uniqueContributors = new Set();
    const totalOperators = Object.keys(mockSubmissionsByOperator).length;
    
    for (const operatorKey in mockSubmissionsByOperator) {
        const submissions = mockSubmissionsByOperator[operatorKey];
        totalSubmissions += submissions.length;
        submissions.forEach(submission => {
            uniqueContributors.add(submission.author);
        });
    }
    
    // Update hero stats
    const statNumbers = document.querySelectorAll('.stat-number');
    if (statNumbers.length >= 3) {
        statNumbers[0].textContent = totalSubmissions;
        statNumbers[1].textContent = uniqueContributors.size;
        statNumbers[2].textContent = totalOperators;
    }
}

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Loading state simulation
function showLoading(element) {
    element.classList.add('loading');
}

function hideLoading(element) {
    element.classList.remove('loading');
}

// Animation on scroll
function animateOnScroll() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.submission-card, .leaderboard-table tr').forEach(el => {
        observer.observe(el);
    });
}

// Initialize animations
document.addEventListener('DOMContentLoaded', animateOnScroll);

// Global functions for operator cards and interactions
window.viewOperatorLeaderboard = viewOperatorLeaderboard;
window.backToOperators = backToOperators;
window.showSubmissionDetails = showSubmissionDetails;
window.downloadSubmission = downloadSubmission;