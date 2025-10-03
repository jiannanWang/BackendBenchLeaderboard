import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';

const ModelEvaluationPage = () => {
    const navigate = useNavigate();
    const [selectedModel, setSelectedModel] = useState('all');

    // Mock data for different models and their training submissions
    const models = {
        nanoGPT: {
            title: 'nanoGPT',
            description: 'Small-scale GPT implementation for language modeling',
            icon: 'fas fa-brain',
            submissions: [
                {
                    id: 1,
                    name: 'Optimized Attention Kernel Set',
                    author: 'gpu_ninja',
                    operators: ['Attention', 'Linear', 'LayerNorm', 'Softmax'],
                    totalEpochs: 100,
                    finalLoss: 3.24,
                    avgTimePerEpoch: 12.5,
                    date: '2024-10-01',
                    convergence: 'Fast',
                    efficiency: 92
                },
                {
                    id: 2,
                    name: 'Flash Attention + Fused MLP',
                    author: 'ml_expert',
                    operators: ['FlashAttention', 'FusedMLP', 'RMSNorm'],
                    totalEpochs: 100,
                    finalLoss: 3.18,
                    avgTimePerEpoch: 9.8,
                    date: '2024-09-28',
                    convergence: 'Very Fast',
                    efficiency: 95
                },
                {
                    id: 3,
                    name: 'Baseline PyTorch Implementation',
                    author: 'baseline_user',
                    operators: ['Standard Attention', 'Linear', 'LayerNorm'],
                    totalEpochs: 100,
                    finalLoss: 3.45,
                    avgTimePerEpoch: 18.2,
                    date: '2024-09-25',
                    convergence: 'Slow',
                    efficiency: 78
                }
            ]
        },
        resnet: {
            title: 'ResNet-50',
            description: 'Deep residual network for image classification',
            icon: 'fas fa-image',
            submissions: [
                {
                    id: 4,
                    name: 'Optimized Conv2D + BatchNorm',
                    author: 'conv_master',
                    operators: ['Conv2D', 'BatchNorm', 'ReLU', 'MaxPool'],
                    totalEpochs: 90,
                    finalLoss: 0.82,
                    avgTimePerEpoch: 45.3,
                    date: '2024-10-02',
                    convergence: 'Fast',
                    efficiency: 88
                },
                {
                    id: 5,
                    name: 'Fused Convolution Kernels',
                    author: 'fusion_pro',
                    operators: ['FusedConv', 'FusedBatchNorm', 'ActivationFusion'],
                    totalEpochs: 90,
                    finalLoss: 0.79,
                    avgTimePerEpoch: 38.7,
                    date: '2024-09-30',
                    convergence: 'Very Fast',
                    efficiency: 93
                },
                {
                    id: 6,
                    name: 'Mixed Precision Training',
                    author: 'precision_guru',
                    operators: ['AMP_Conv2D', 'AMP_Linear', 'GradScaler'],
                    totalEpochs: 90,
                    finalLoss: 0.85,
                    avgTimePerEpoch: 41.2,
                    date: '2024-09-27',
                    convergence: 'Fast',
                    efficiency: 90
                }
            ]
        },
        transformer: {
            title: 'Vision Transformer',
            description: 'Transformer architecture for computer vision tasks',
            icon: 'fas fa-eye',
            submissions: [
                {
                    id: 7,
                    name: 'Efficient Multi-Head Attention',
                    author: 'attention_wizard',
                    operators: ['MultiHeadAttention', 'PatchEmbedding', 'PositionalEncoding'],
                    totalEpochs: 300,
                    finalLoss: 1.12,
                    avgTimePerEpoch: 28.4,
                    date: '2024-10-01',
                    convergence: 'Medium',
                    efficiency: 85
                },
                {
                    id: 8,
                    name: 'Sparse Attention Patterns',
                    author: 'sparse_expert',
                    operators: ['SparseAttention', 'EfficientFFN', 'DynamicPositions'],
                    totalEpochs: 300,
                    finalLoss: 1.08,
                    avgTimePerEpoch: 22.1,
                    date: '2024-09-29',
                    convergence: 'Fast',
                    efficiency: 91
                }
            ]
        },
        lstm: {
            title: 'LSTM Language Model',
            description: 'Recurrent neural network for sequence modeling',
            icon: 'fas fa-stream',
            submissions: [
                {
                    id: 9,
                    name: 'Optimized LSTM Cells',
                    author: 'rnn_ninja',
                    operators: ['LSTMCell', 'Embedding', 'Dense'],
                    totalEpochs: 50,
                    finalLoss: 4.23,
                    avgTimePerEpoch: 33.7,
                    date: '2024-09-26',
                    convergence: 'Slow',
                    efficiency: 82
                },
                {
                    id: 10,
                    name: 'Fused LSTM Operations',
                    author: 'sequence_pro',
                    operators: ['FusedLSTM', 'FastEmbedding', 'OptimizedDense'],
                    totalEpochs: 50,
                    finalLoss: 4.11,
                    avgTimePerEpoch: 26.9,
                    date: '2024-09-24',
                    convergence: 'Medium',
                    efficiency: 89
                }
            ]
        }
    };

    const filteredModels = selectedModel === 'all' ? 
        Object.entries(models) : 
        Object.entries(models).filter(([key]) => key === selectedModel);

    const getConvergenceColor = (convergence) => {
        switch (convergence) {
            case 'Very Fast': return '#00ff00';
            case 'Fast': return '#00d9ff';
            case 'Medium': return '#ffc107';
            case 'Slow': return '#ff6b6b';
            default: return '#888';
        }
    };

    const handleViewTrainingCurve = (modelKey, submission) => {
        navigate(`/model-evaluation/${modelKey}/training/${submission.id}`);
    };

    return (
        <div className="model-evaluation-page">
            {/* Header */}
            <section className="page-header model-eval-header">
                <div className="container">
                    <div className="header-content">
                        <div className="header-text">
                            <h1><i className="fas fa-chart-line"></i> Model Evaluation</h1>
                            <p>Compare training performance across different operator implementations. 
                               Evaluate how kernel optimizations affect model convergence, training time, and final accuracy.</p>
                        </div>
                        <div className="header-stats">
                            <div className="stat">
                                <span className="stat-number">{Object.values(models).reduce((sum, model) => sum + model.submissions.length, 0)}</span>
                                <span className="stat-label">Training Runs</span>
                            </div>
                            <div className="stat">
                                <span className="stat-number">{Object.keys(models).length}</span>
                                <span className="stat-label">Model Types</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Model Filter */}
            <section className="model-filter-section">
                <div className="container">
                    <div className="filter-controls">
                        <h3>Filter by Model Type</h3>
                        <div className="model-filter-buttons">
                            <button 
                                className={`filter-btn ${selectedModel === 'all' ? 'active' : ''}`}
                                onClick={() => setSelectedModel('all')}
                            >
                                <i className="fas fa-th-large"></i> All Models
                            </button>
                            {Object.entries(models).map(([key, model]) => (
                                <button 
                                    key={key}
                                    className={`filter-btn ${selectedModel === key ? 'active' : ''}`}
                                    onClick={() => setSelectedModel(key)}
                                >
                                    <i className={model.icon}></i> {model.title}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>
            </section>

            {/* Models Grid */}
            <section className="models-section">
                <div className="container">
                    <div className="models-grid">
                        {filteredModels.map(([modelKey, model]) => (
                            <div key={modelKey} className="model-block">
                                <div className="model-header">
                                    <div className="model-title">
                                        <i className={model.icon}></i>
                                        <div>
                                            <h3>{model.title}</h3>
                                            <p>{model.description}</p>
                                        </div>
                                    </div>
                                    <div className="model-stats">
                                        <span className="submission-count">{model.submissions.length} runs</span>
                                    </div>
                                </div>

                                <div className="submissions-list">
                                    {model.submissions.map((submission) => (
                                        <div key={submission.id} className="submission-card">
                                            <div className="submission-info">
                                                <div className="submission-header">
                                                    <h4>{submission.name}</h4>
                                                    <span className="author">by @{submission.author}</span>
                                                </div>
                                                
                                                <div className="operators-used">
                                                    <span className="operators-label">Operators:</span>
                                                    <div className="operators-tags">
                                                        {submission.operators.map((op, idx) => (
                                                            <span key={idx} className="operator-tag">{op}</span>
                                                        ))}
                                                    </div>
                                                </div>

                                                <div className="training-metrics">
                                                    <div className="metric">
                                                        <span className="metric-label">Final Loss</span>
                                                        <span className="metric-value loss">{submission.finalLoss}</span>
                                                    </div>
                                                    <div className="metric">
                                                        <span className="metric-label">Avg Time/Epoch</span>
                                                        <span className="metric-value time">{submission.avgTimePerEpoch}s</span>
                                                    </div>
                                                    <div className="metric">
                                                        <span className="metric-label">Convergence</span>
                                                        <span 
                                                            className="metric-value convergence"
                                                            style={{ color: getConvergenceColor(submission.convergence) }}
                                                        >
                                                            {submission.convergence}
                                                        </span>
                                                    </div>
                                                    <div className="metric">
                                                        <span className="metric-label">Efficiency</span>
                                                        <span className="metric-value efficiency">{submission.efficiency}%</span>
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="submission-actions">
                                                <button 
                                                    className="action-btn primary"
                                                    onClick={() => handleViewTrainingCurve(modelKey, submission)}
                                                >
                                                    <i className="fas fa-chart-line"></i> View Training Curve
                                                </button>
                                                <button className="action-btn secondary">
                                                    <i className="fas fa-download"></i>
                                                </button>
                                                <button className="action-btn secondary">
                                                    <i className="fas fa-code"></i>
                                                </button>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </section>

            {/* Quick Compare Section */}
            <section className="quick-compare-section">
                <div className="container">
                    <div className="section-header">
                        <h2><i className="fas fa-balance-scale"></i> Quick Performance Comparison</h2>
                        <p>Compare key metrics across all submissions</p>
                    </div>
                    
                    <div className="comparison-grid">
                        <div className="comparison-chart">
                            <h4>Training Time Efficiency</h4>
                            <div className="chart-placeholder">
                                <div className="chart-bars">
                                    {Object.values(models).flatMap(model => 
                                        model.submissions.slice(0, 2) // Show top 2 from each model
                                    ).sort((a, b) => a.avgTimePerEpoch - b.avgTimePerEpoch).slice(0, 6).map((submission, idx) => (
                                        <div key={submission.id} className="chart-bar">
                                            <div className="bar-label">{submission.name.substring(0, 20)}...</div>
                                            <div className="bar-container">
                                                <div 
                                                    className="bar-fill time-bar" 
                                                    style={{ 
                                                        width: `${100 - (submission.avgTimePerEpoch / 50) * 100}%` 
                                                    }}
                                                ></div>
                                            </div>
                                            <div className="bar-value">{submission.avgTimePerEpoch}s</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="comparison-chart">
                            <h4>Final Loss Achievement</h4>
                            <div className="chart-placeholder">
                                <div className="chart-bars">
                                    {Object.values(models).flatMap(model => 
                                        model.submissions.slice(0, 2) // Show top 2 from each model
                                    ).sort((a, b) => a.finalLoss - b.finalLoss).slice(0, 6).map((submission, idx) => (
                                        <div key={submission.id} className="chart-bar">
                                            <div className="bar-label">{submission.name.substring(0, 20)}...</div>
                                            <div className="bar-container">
                                                <div 
                                                    className="bar-fill loss-bar" 
                                                    style={{ 
                                                        width: `${Math.max(20, 100 - (submission.finalLoss / 5) * 80)}%` 
                                                    }}
                                                ></div>
                                            </div>
                                            <div className="bar-value">{submission.finalLoss}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default ModelEvaluationPage;