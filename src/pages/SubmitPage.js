import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const SubmitPage = () => {
    const [uploadedFiles, setUploadedFiles] = useState([]);
    const [formData, setFormData] = useState({
        kernelName: '',
        operatorType: '',
        dslType: '',
        targetDevice: '',
        description: '',
        authorName: '',
        authorEmail: '',
        precision: '',
        batchSize: '',
        githubRepo: '',
        additionalNotes: ''
    });

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleFileUpload = (files) => {
        const newFiles = Array.from(files).filter(file => 
            !uploadedFiles.some(f => f.name === file.name)
        );
        setUploadedFiles(prev => [...prev, ...newFiles]);
    };

    const removeFile = (fileName) => {
        setUploadedFiles(prev => prev.filter(f => f.name !== fileName));
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    };

    const handleDragLeave = (e) => {
        e.currentTarget.classList.remove('dragover');
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        handleFileUpload(e.dataTransfer.files);
    };

    const handleSubmit = (e) => {
        e.preventDefault();
        
        // Validate required fields
        const requiredFields = ['kernelName', 'operatorType', 'dslType', 'targetDevice', 'description', 'authorName', 'authorEmail'];
        const missingFields = requiredFields.filter(field => !formData[field].trim());
        
        if (missingFields.length > 0) {
            alert('Please fill in all required fields.');
            return;
        }
        
        if (uploadedFiles.length === 0) {
            alert('Please upload at least one kernel file.');
            return;
        }
        
        // Simulate submission
        alert('Your kernel submission has been received! You will receive an email confirmation shortly.');
    };

    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    return (
        <div>
            {/* Hero Section */}
            <section className="hero">
                <div className="container">
                    <div className="hero-content">
                        <h2>Submit Your GPU Kernel</h2>
                        <p>Share your optimized kernel implementations with the research community. Get automated benchmarking and performance comparison.</p>
                    </div>
                </div>
            </section>

            {/* Submission Form */}
            <section className="leaderboard">
                <div className="container">
                    {/* Requirements */}
                    <div className="requirements">
                        <h4>
                            <i className="fas fa-info-circle"></i>
                            Submission Requirements
                        </h4>
                        <ul>
                            <li>Kernel must implement one of the supported operators (Matrix Multiplication, Attention, Layer Normalization, etc.)</li>
                            <li>Code should be well-documented with clear function signatures</li>
                            <li>Include test cases that verify correctness</li>
                            <li>Provide performance benchmarking script (optional but recommended)</li>
                            <li>Ensure code compiles and runs on the specified target device</li>
                            <li>Maximum file size: 50MB per submission</li>
                        </ul>
                    </div>

                    {/* Submission Form */}
                    <form className="submission-form" onSubmit={handleSubmit}>
                        <h3>Kernel Information</h3>
                        
                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="kernel-name">Kernel Name *</label>
                                <input 
                                    type="text" 
                                    id="kernel-name" 
                                    name="kernelName" 
                                    value={formData.kernelName}
                                    onChange={handleInputChange}
                                    required 
                                    placeholder="e.g., OptimizedMatMul_v3"
                                />
                            </div>
                            <div className="form-group">
                                <label htmlFor="operator-type">Operator Type *</label>
                                <select 
                                    id="operator-type" 
                                    name="operatorType" 
                                    value={formData.operatorType}
                                    onChange={handleInputChange}
                                    required
                                >
                                    <option value="">Select Operator</option>
                                    <option value="matmul">Matrix Multiplication</option>
                                    <option value="conv2d">2D Convolution</option>
                                    <option value="layernorm">Layer Normalization</option>
                                    <option value="attention">Attention</option>
                                    <option value="softmax">Softmax</option>
                                    <option value="embedding">Embedding</option>
                                    <option value="linear">Linear</option>
                                    <option value="gelu">GELU</option>
                                </select>
                            </div>
                        </div>

                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="dsl-type">DSL/Framework *</label>
                                <select 
                                    id="dsl-type" 
                                    name="dslType" 
                                    value={formData.dslType}
                                    onChange={handleInputChange}
                                    required
                                >
                                    <option value="">Select DSL</option>
                                    <option value="triton">Triton</option>
                                    <option value="cuda">CUDA</option>
                                    <option value="cutlass">CUTLASS</option>
                                    <option value="pytorch">PyTorch</option>
                                    <option value="jax">JAX</option>
                                    <option value="tvm">TVM</option>
                                    <option value="other">Other</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label htmlFor="target-device">Target Device *</label>
                                <select 
                                    id="target-device" 
                                    name="targetDevice" 
                                    value={formData.targetDevice}
                                    onChange={handleInputChange}
                                    required
                                >
                                    <option value="">Select Device</option>
                                    <option value="h100">NVIDIA H100</option>
                                    <option value="a100">NVIDIA A100</option>
                                    <option value="v100">NVIDIA V100</option>
                                    <option value="rtx4090">NVIDIA RTX 4090</option>
                                    <option value="rtx3080">NVIDIA RTX 3080</option>
                                    <option value="other">Other</option>
                                </select>
                            </div>
                        </div>

                        <div className="form-group">
                            <label htmlFor="description">Description *</label>
                            <textarea 
                                id="description" 
                                name="description" 
                                value={formData.description}
                                onChange={handleInputChange}
                                required 
                                placeholder="Describe your kernel implementation, optimizations, and key features..."
                            />
                        </div>

                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="author-name">Author Name *</label>
                                <input 
                                    type="text" 
                                    id="author-name" 
                                    name="authorName" 
                                    value={formData.authorName}
                                    onChange={handleInputChange}
                                    required 
                                    placeholder="Your name or handle"
                                />
                            </div>
                            <div className="form-group">
                                <label htmlFor="author-email">Email *</label>
                                <input 
                                    type="email" 
                                    id="author-email" 
                                    name="authorEmail" 
                                    value={formData.authorEmail}
                                    onChange={handleInputChange}
                                    required 
                                    placeholder="your.email@example.com"
                                />
                            </div>
                        </div>

                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="precision">Precision</label>
                                <select 
                                    id="precision" 
                                    name="precision"
                                    value={formData.precision}
                                    onChange={handleInputChange}
                                >
                                    <option value="">Select Precision</option>
                                    <option value="fp16">FP16</option>
                                    <option value="fp32">FP32</option>
                                    <option value="bf16">BF16</option>
                                    <option value="int8">INT8</option>
                                    <option value="mixed">Mixed Precision</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label htmlFor="batch-size">Batch Size</label>
                                <input 
                                    type="number" 
                                    id="batch-size" 
                                    name="batchSize" 
                                    value={formData.batchSize}
                                    onChange={handleInputChange}
                                    placeholder="e.g., 1024"
                                />
                            </div>
                        </div>

                        <h3>File Upload</h3>
                        
                        <div className="form-group">
                            <label>Kernel Files *</label>
                            <div 
                                className="file-upload"
                                onDragOver={handleDragOver}
                                onDragLeave={handleDragLeave}
                                onDrop={handleDrop}
                                onClick={() => document.getElementById('file-input').click()}
                            >
                                <input 
                                    type="file" 
                                    id="file-input" 
                                    multiple 
                                    accept=".py,.cu,.cpp,.h,.hpp,.c,.txt,.md"
                                    onChange={(e) => handleFileUpload(e.target.files)}
                                    style={{ display: 'none' }}
                                />
                                <div className="file-upload-content">
                                    <div className="file-upload-icon">
                                        <i className="fas fa-cloud-upload-alt"></i>
                                    </div>
                                    <p>Drag and drop your files here, or <strong>click to browse</strong></p>
                                    <p>Supported formats: .py, .cu, .cpp, .h, .hpp, .c, .txt, .md</p>
                                </div>
                            </div>
                            <div className="file-list">
                                {uploadedFiles.map((file, index) => (
                                    <div key={index} className="file-item">
                                        <div className="file-item-info">
                                            <span className="file-item-name">{file.name}</span>
                                            <span className="file-item-size">({formatFileSize(file.size)})</span>
                                        </div>
                                        <button 
                                            type="button" 
                                            className="file-item-remove" 
                                            onClick={() => removeFile(file.name)}
                                        >
                                            <i className="fas fa-times"></i>
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="form-group">
                            <label htmlFor="github-repo">GitHub Repository (Optional)</label>
                            <input 
                                type="url" 
                                id="github-repo" 
                                name="githubRepo" 
                                value={formData.githubRepo}
                                onChange={handleInputChange}
                                placeholder="https://github.com/username/repository"
                            />
                        </div>

                        <div className="form-group">
                            <label htmlFor="additional-notes">Additional Notes</label>
                            <textarea 
                                id="additional-notes" 
                                name="additionalNotes" 
                                value={formData.additionalNotes}
                                onChange={handleInputChange}
                                placeholder="Any additional information, dependencies, or special instructions..."
                            />
                        </div>

                        <div className="submit-section">
                            <button type="button" className="btn btn-draft">Save as Draft</button>
                            <button type="submit" className="btn btn-submit">
                                <i className="fas fa-rocket"></i> Submit for Review
                            </button>
                        </div>
                    </form>
                </div>
            </section>
        </div>
    );
};

export default SubmitPage;