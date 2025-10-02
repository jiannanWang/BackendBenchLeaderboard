import React, { useEffect } from 'react';

const SubmissionModal = ({ submission, onClose }) => {
    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape') {
                onClose();
            }
        };

        document.addEventListener('keydown', handleEscape);
        document.body.style.overflow = 'hidden';

        return () => {
            document.removeEventListener('keydown', handleEscape);
            document.body.style.overflow = 'auto';
        };
    }, [onClose]);

    const handleModalClick = (e) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    return (
        <div className="modal show" onClick={handleModalClick}>
            <div className="modal-content">
                <div className="modal-header">
                    <h3>{submission.name}</h3>
                    <button className="modal-close" onClick={onClose}>
                        <i className="fas fa-times"></i>
                    </button>
                </div>
                <div className="modal-body">
                    <div className="submission-details">
                        <div className="detail-section">
                            <h4>Performance Metrics</h4>
                            <div className="metrics-grid">
                                <div className="metric">
                                    <span className="metric-label">TFLOPS</span>
                                    <span className="metric-value">{submission.performance.toLocaleString()}</span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Latency</span>
                                    <span className="metric-value">{submission.latency}ms</span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Memory BW</span>
                                    <span className="metric-value">{submission.memoryBW}TB/s</span>
                                </div>
                                <div className="metric">
                                    <span className="metric-label">Efficiency</span>
                                    <span className="metric-value">{submission.efficiency}%</span>
                                </div>
                            </div>
                        </div>
                        <div className="detail-section">
                            <h4>Configuration</h4>
                            <div className="config-table">
                                <div className="config-row">
                                    <span className="config-key">Operator</span>
                                    <span className="config-value">{submission.operator}</span>
                                </div>
                                <div className="config-row">
                                    <span className="config-key">DSL</span>
                                    <span className="config-value">{submission.dsl}</span>
                                </div>
                                <div className="config-row">
                                    <span className="config-key">Device</span>
                                    <span className="config-value">NVIDIA {submission.device}</span>
                                </div>
                                <div className="config-row">
                                    <span className="config-key">Precision</span>
                                    <span className="config-value">{submission.precision}</span>
                                </div>
                                <div className="config-row">
                                    <span className="config-key">Batch Size</span>
                                    <span className="config-value">{submission.batchSize}</span>
                                </div>
                            </div>
                        </div>
                        <div className="detail-section">
                            <h4>Submission Info</h4>
                            <div className="submission-meta-full">
                                <div className="meta-item">
                                    <span className="meta-label">Author</span>
                                    <span className="meta-value">@{submission.author}</span>
                                </div>
                                <div className="meta-item">
                                    <span className="meta-label">Submitted</span>
                                    <span className="meta-value">
                                        {new Date(submission.date).toLocaleDateString('en-US', { 
                                            year: 'numeric', 
                                            month: 'long', 
                                            day: 'numeric' 
                                        })}
                                    </span>
                                </div>
                                <div className="meta-item">
                                    <span className="meta-label">Submission ID</span>
                                    <span className="meta-value">#{submission.id}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="modal-footer">
                    <button className="btn-secondary">View Code</button>
                    <button className="btn-primary">Download Kernel</button>
                </div>
            </div>
        </div>
    );
};

export default SubmissionModal;