import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';

const TrainingResultPage = () => {
    const { modelKey, submissionId } = useParams();
    const navigate = useNavigate();
    const [trainingData, setTrainingData] = useState(null);
    const [activeChart, setActiveChart] = useState('loss');

    // Mock detailed training data
    const mockTrainingData = {
        1: { // nanoGPT - Optimized Attention Kernel Set
            submission: {
                id: 1,
                name: 'Optimized Attention Kernel Set',
                author: 'gpu_ninja',
                model: 'nanoGPT',
                operators: ['Attention', 'Linear', 'LayerNorm', 'Softmax'],
                totalEpochs: 100,
                finalLoss: 3.24,
                avgTimePerEpoch: 12.5,
                date: '2024-10-01',
                convergence: 'Fast',
                efficiency: 92
            },
            trainingLoss: Array.from({length: 100}, (_, i) => {
                const epoch = i + 1;
                const base = 6.5 * Math.exp(-epoch / 30) + 3.2;
                const noise = (Math.random() - 0.5) * 0.1;
                return Math.max(3.15, base + noise);
            }),
            validationLoss: Array.from({length: 100}, (_, i) => {
                const epoch = i + 1;
                const base = 6.8 * Math.exp(-epoch / 32) + 3.3;
                const noise = (Math.random() - 0.5) * 0.12;
                return Math.max(3.2, base + noise);
            }),
            timePerEpoch: Array.from({length: 100}, (_, i) => {
                const base = 12.5 + (Math.random() - 0.5) * 2;
                return Math.max(10, base);
            }),
            memoryUsage: Array.from({length: 100}, (_, i) => {
                return 8.2 + (Math.random() - 0.5) * 0.5;
            }),
            learningRate: Array.from({length: 100}, (_, i) => {
                const epoch = i + 1;
                if (epoch <= 10) return 0.0001;
                if (epoch <= 30) return 0.0003;
                if (epoch <= 70) return 0.0001;
                return 0.00005;
            })
        },
        2: { // nanoGPT - Flash Attention + Fused MLP
            submission: {
                id: 2,
                name: 'Flash Attention + Fused MLP',
                author: 'ml_expert',
                model: 'nanoGPT',
                operators: ['FlashAttention', 'FusedMLP', 'RMSNorm'],
                totalEpochs: 100,
                finalLoss: 3.18,
                avgTimePerEpoch: 9.8,
                date: '2024-09-28',
                convergence: 'Very Fast',
                efficiency: 95
            },
            trainingLoss: Array.from({length: 100}, (_, i) => {
                const epoch = i + 1;
                const base = 6.2 * Math.exp(-epoch / 25) + 3.1;
                const noise = (Math.random() - 0.5) * 0.08;
                return Math.max(3.1, base + noise);
            }),
            validationLoss: Array.from({length: 100}, (_, i) => {
                const epoch = i + 1;
                const base = 6.4 * Math.exp(-epoch / 27) + 3.2;
                const noise = (Math.random() - 0.5) * 0.1;
                return Math.max(3.15, base + noise);
            }),
            timePerEpoch: Array.from({length: 100}, (_, i) => {
                const base = 9.8 + (Math.random() - 0.5) * 1.5;
                return Math.max(8, base);
            }),
            memoryUsage: Array.from({length: 100}, (_, i) => {
                return 6.8 + (Math.random() - 0.5) * 0.4;
            }),
            learningRate: Array.from({length: 100}, (_, i) => {
                const epoch = i + 1;
                if (epoch <= 10) return 0.0001;
                if (epoch <= 30) return 0.0003;
                if (epoch <= 70) return 0.0001;
                return 0.00005;
            })
        }
        // Add more submissions as needed
    };

    useEffect(() => {
        const data = mockTrainingData[submissionId];
        if (data) {
            setTrainingData(data);
        }
    }, [submissionId]);

    if (!trainingData) {
        return (
            <div className="container" style={{ padding: '3rem 0' }}>
                <h2>Training data not found</h2>
                <Link to="/model-evaluation">Back to Model Evaluation</Link>
            </div>
        );
    }

    const { submission, trainingLoss, validationLoss, timePerEpoch, memoryUsage, learningRate } = trainingData;

    const renderChart = (data, label, color, yAxisLabel, formatValue = (v) => v.toFixed(3)) => {
        const maxValue = Math.max(...data);
        const minValue = Math.min(...data);
        const range = maxValue - minValue;
        
        return (
            <div className="training-chart">
                <div className="chart-header">
                    <h4>{label}</h4>
                    <div className="chart-stats">
                        <span>Min: {formatValue(minValue)}</span>
                        <span>Max: {formatValue(maxValue)}</span>
                        <span>Final: {formatValue(data[data.length - 1])}</span>
                    </div>
                </div>
                <div className="chart-container">
                    <div className="chart-y-axis">
                        <span className="y-axis-label">{yAxisLabel}</span>
                        <div className="y-axis-ticks">
                            <span>{formatValue(maxValue)}</span>
                            <span>{formatValue(minValue + range * 0.75)}</span>
                            <span>{formatValue(minValue + range * 0.5)}</span>
                            <span>{formatValue(minValue + range * 0.25)}</span>
                            <span>{formatValue(minValue)}</span>
                        </div>
                    </div>
                    <div className="chart-area">
                        <svg viewBox="0 0 800 300" className="chart-svg">
                            <defs>
                                <linearGradient id={`gradient-${color}`} x1="0%" y1="0%" x2="0%" y2="100%">
                                    <stop offset="0%" style={{ stopColor: color, stopOpacity: 0.3 }} />
                                    <stop offset="100%" style={{ stopColor: color, stopOpacity: 0.05 }} />
                                </linearGradient>
                            </defs>
                            
                            {/* Grid lines */}
                            {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
                                <line
                                    key={ratio}
                                    x1="0"
                                    y1={300 - ratio * 300}
                                    x2="800"
                                    y2={300 - ratio * 300}
                                    stroke="rgba(255,255,255,0.1)"
                                    strokeWidth="1"
                                />
                            ))}
                            
                            {/* Vertical grid lines */}
                            {[0, 0.25, 0.5, 0.75, 1].map((ratio) => (
                                <line
                                    key={ratio}
                                    x1={ratio * 800}
                                    y1="0"
                                    x2={ratio * 800}
                                    y2="300"
                                    stroke="rgba(255,255,255,0.1)"
                                    strokeWidth="1"
                                />
                            ))}
                            
                            {/* Area under curve */}
                            <path
                                d={`M 0,300 ${data.map((value, index) => {
                                    const x = (index / (data.length - 1)) * 800;
                                    const y = 300 - ((value - minValue) / range) * 300;
                                    return `L ${x},${y}`;
                                }).join(' ')} L 800,300 Z`}
                                fill={`url(#gradient-${color})`}
                            />
                            
                            {/* Main line */}
                            <path
                                d={`M ${data.map((value, index) => {
                                    const x = (index / (data.length - 1)) * 800;
                                    const y = 300 - ((value - minValue) / range) * 300;
                                    return `${x},${y}`;
                                }).join(' L ')}`}
                                fill="none"
                                stroke={color}
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            />
                            
                            {/* Data points */}
                            {data.filter((_, i) => i % 10 === 0).map((value, index) => {
                                const actualIndex = index * 10;
                                const x = (actualIndex / (data.length - 1)) * 800;
                                const y = 300 - ((value - minValue) / range) * 300;
                                return (
                                    <circle
                                        key={actualIndex}
                                        cx={x}
                                        cy={y}
                                        r="3"
                                        fill={color}
                                        stroke="white"
                                        strokeWidth="1"
                                    />
                                );
                            })}
                        </svg>
                        
                        {/* X-axis */}
                        <div className="chart-x-axis">
                            <span>0</span>
                            <span>{Math.floor(data.length * 0.25)}</span>
                            <span>{Math.floor(data.length * 0.5)}</span>
                            <span>{Math.floor(data.length * 0.75)}</span>
                            <span>{data.length}</span>
                        </div>
                        <div className="x-axis-label">Epochs</div>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="training-result-page">
            {/* Header */}
            <section className="training-header">
                <div className="container">
                    <div className="training-nav">
                        <button 
                            onClick={() => navigate(-1)} 
                            className="back-btn"
                        >
                            <i className="fas fa-arrow-left"></i> Back to Model Evaluation
                        </button>
                        <div className="breadcrumb">
                            <Link to="/">Home</Link>
                            <span className="separator">/</span>
                            <Link to="/model-evaluation">Model Evaluation</Link>
                            <span className="separator">/</span>
                            <span>{submission.model}</span>
                            <span className="separator">/</span>
                            <span>{submission.name}</span>
                        </div>
                    </div>

                    <div className="training-info">
                        <div className="training-title">
                            <h1>{submission.name}</h1>
                            <div className="training-badges">
                                <span className="model-badge">{submission.model}</span>
                                <span className="author-badge">@{submission.author}</span>
                            </div>
                        </div>

                        <div className="operators-used">
                            <h3>Operators Used:</h3>
                            <div className="operator-tags">
                                {submission.operators.map((op, idx) => (
                                    <span key={idx} className="operator-tag">{op}</span>
                                ))}
                            </div>
                        </div>

                        <div className="training-summary-stats">
                            <div className="summary-stat">
                                <div className="stat-icon"><i className="fas fa-chart-line"></i></div>
                                <div className="stat-content">
                                    <span className="stat-value">{submission.finalLoss}</span>
                                    <span className="stat-label">Final Loss</span>
                                </div>
                            </div>
                            <div className="summary-stat">
                                <div className="stat-icon"><i className="fas fa-clock"></i></div>
                                <div className="stat-content">
                                    <span className="stat-value">{submission.avgTimePerEpoch}s</span>
                                    <span className="stat-label">Avg Time/Epoch</span>
                                </div>
                            </div>
                            <div className="summary-stat">
                                <div className="stat-icon"><i className="fas fa-layer-group"></i></div>
                                <div className="stat-content">
                                    <span className="stat-value">{submission.totalEpochs}</span>
                                    <span className="stat-label">Total Epochs</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Chart Selection */}
            <section className="chart-controls">
                <div className="container">
                    <div className="chart-tabs">
                        <button 
                            className={`chart-tab ${activeChart === 'loss' ? 'active' : ''}`}
                            onClick={() => setActiveChart('loss')}
                        >
                            <i className="fas fa-chart-line"></i> Loss Curves
                        </button>
                        <button 
                            className={`chart-tab ${activeChart === 'time' ? 'active' : ''}`}
                            onClick={() => setActiveChart('time')}
                        >
                            <i className="fas fa-clock"></i> Training Time
                        </button>
                        <button 
                            className={`chart-tab ${activeChart === 'memory' ? 'active' : ''}`}
                            onClick={() => setActiveChart('memory')}
                        >
                            <i className="fas fa-memory"></i> Memory Usage
                        </button>
                    </div>
                </div>
            </section>

            {/* Charts */}
            <section className="charts-section">
                <div className="container">
                    {activeChart === 'loss' && (
                        <div className="charts-grid">
                            <div className="chart-wrapper">
                                {renderChart(trainingLoss, 'Training Loss', '#00d9ff', 'Loss')}
                            </div>
                            <div className="chart-wrapper">
                                {renderChart(validationLoss, 'Validation Loss', '#ff6b6b', 'Loss')}
                            </div>
                        </div>
                    )}

                    {activeChart === 'time' && (
                        <div className="charts-grid single">
                            <div className="chart-wrapper">
                                {renderChart(timePerEpoch, 'Time per Epoch', '#ffc107', 'Time (seconds)', (v) => `${v.toFixed(1)}s`)}
                            </div>
                        </div>
                    )}

                    {activeChart === 'memory' && (
                        <div className="charts-grid single">
                            <div className="chart-wrapper">
                                {renderChart(memoryUsage, 'Memory Usage', '#4caf50', 'Memory (GB)', (v) => `${v.toFixed(1)}GB`)}
                            </div>
                        </div>
                    )}

                    {activeChart === 'lr' && (
                        <div className="charts-grid single">
                            <div className="chart-wrapper">
                                {renderChart(learningRate, 'Learning Rate Schedule', '#9c27b0', 'Learning Rate', (v) => v.toExponential(2))}
                            </div>
                        </div>
                    )}
                </div>
            </section>

            {/* Training Insights */}
            <section className="training-insights">
                <div className="container">
                    <h2><i className="fas fa-lightbulb"></i> Training Insights</h2>
                    <div className="insights-grid">
                        <div className="insight-card performance">
                            <h4>Performance Analysis</h4>
                            <ul>
                                <li>Achieved {((1 - submission.finalLoss / 6.5) * 100).toFixed(1)}% improvement over baseline loss</li>
                                <li>Average epoch time of {submission.avgTimePerEpoch}s represents {submission.efficiency}% efficiency</li>
                                <li>Convergence rate classified as: <strong>{submission.convergence}</strong></li>
                                <li>Memory efficiency optimized through {submission.operators.join(', ')} operators</li>
                            </ul>
                        </div>

                        <div className="insight-card optimization">
                            <h4>Optimization Highlights</h4>
                            <ul>
                                {submission.operators.includes('FlashAttention') && (
                                    <li>FlashAttention reduces memory complexity from O(N²) to O(N)</li>
                                )}
                                {submission.operators.includes('FusedMLP') && (
                                    <li>Fused MLP operations eliminate intermediate activations</li>
                                )}
                                {submission.operators.includes('RMSNorm') && (
                                    <li>RMSNorm provides faster normalization than LayerNorm</li>
                                )}
                                {submission.operators.includes('Attention') && (
                                    <li>Optimized attention kernels improve sequence processing</li>
                                )}
                                <li>Kernel fusion reduces GPU memory bandwidth requirements</li>
                                <li>Mixed precision training accelerates convergence</li>
                            </ul>
                        </div>

                        <div className="insight-card comparison">
                            <h4>Compared to Baseline</h4>
                            <div className="comparison-metrics">
                                <div className="comparison-item">
                                    <span className="comparison-label">Training Speed</span>
                                    <div className="comparison-bar">
                                        <div 
                                            className="comparison-fill faster" 
                                            style={{ width: `${Math.min(100, (18.2 / submission.avgTimePerEpoch - 1) * 100)}%` }}
                                        ></div>
                                    </div>
                                    <span className="comparison-value">
                                        {((18.2 / submission.avgTimePerEpoch - 1) * 100).toFixed(1)}% faster
                                    </span>
                                </div>
                                <div className="comparison-item">
                                    <span className="comparison-label">Final Loss</span>
                                    <div className="comparison-bar">
                                        <div 
                                            className="comparison-fill better" 
                                            style={{ width: `${Math.min(100, (3.45 / submission.finalLoss - 1) * 100)}%` }}
                                        ></div>
                                    </div>
                                    <span className="comparison-value">
                                        {((3.45 / submission.finalLoss - 1) * 100).toFixed(1)}% better
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default TrainingResultPage;