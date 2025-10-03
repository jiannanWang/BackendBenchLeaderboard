import React, { useState, useEffect } from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import LeaderboardTable from '../components/LeaderboardTable';
import { mockSubmissionsByOperator, operatorInfo } from '../data/mockData';

const OperatorPage = () => {
    const { operatorKey } = useParams();
    const navigate = useNavigate();
    const [submissions, setSubmissions] = useState([]);
    const [filteredSubmissions, setFilteredSubmissions] = useState([]);
    const [currentView, setCurrentView] = useState('table');
    const [filters, setFilters] = useState({
        dsl: '',
        device: '',
        sort: 'performance'
    });

    useEffect(() => {
        if (operatorKey && mockSubmissionsByOperator[operatorKey]) {
            const operatorSubmissions = [...mockSubmissionsByOperator[operatorKey]];
            setSubmissions(operatorSubmissions);
            setFilteredSubmissions(operatorSubmissions);
        }
    }, [operatorKey]);

    useEffect(() => {
        applyFiltersAndSort();
    }, [filters, submissions]);

    const applyFiltersAndSort = () => {
        let filtered = [...submissions];

        // Apply filters
        if (filters.dsl) {
            filtered = filtered.filter(sub => 
                sub.dsl.toLowerCase() === filters.dsl.toLowerCase()
            );
        }

        if (filters.device) {
            filtered = filtered.filter(sub => 
                sub.device.toLowerCase().replace(' ', '') === filters.device.toLowerCase()
            );
        }

        // Apply sorting
        filtered.sort((a, b) => {
            switch (filters.sort) {
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

        setFilteredSubmissions(filtered);
    };

    const handleFilterChange = (filterType, value) => {
        setFilters(prev => ({
            ...prev,
            [filterType]: value
        }));
    };

    const toggleView = (view) => {
        setCurrentView(view);
    };

    if (!operatorKey || !operatorInfo[operatorKey]) {
        return (
            <div className="container" style={{ padding: '3rem 0' }}>
                <h2>Operator not found</h2>
                <Link to="/">Back to home</Link>
            </div>
        );
    }

    const info = operatorInfo[operatorKey];

    return (
        <div>
            {/* Individual Operator Leaderboard */}
            <section className="leaderboard" id="leaderboard">
                <div className="container">
                    <div className="leaderboard-nav">
                        <Link to="/" className="back-btn">
                            <i className="fas fa-arrow-left"></i> Back to Operators
                        </Link>
                        <div className="current-operator">
                            <h3 id="current-operator-title">{info.title} Leaderboard</h3>
                            <p id="current-operator-description">
                                Performance rankings for {info.title.toLowerCase()} kernels
                            </p>
                        </div>
                    </div>

                    <div className="leaderboard-controls">
                        <div className="filters-mini">
                            <div className="filter-group">
                                <label>DSL</label>
                                <select 
                                    value={filters.dsl} 
                                    onChange={(e) => handleFilterChange('dsl', e.target.value)}
                                >
                                    <option value="">All DSLs</option>
                                    <option value="triton">Triton</option>
                                    <option value="cuda">CUDA</option>
                                    <option value="cutlass">CUTLASS</option>
                                    <option value="pytorch">PyTorch</option>
                                    <option value="jax">JAX</option>
                                    <option value="tvm">TVM</option>
                                </select>
                            </div>
                            <div className="filter-group">
                                <label>Device</label>
                                <select 
                                    value={filters.device} 
                                    onChange={(e) => handleFilterChange('device', e.target.value)}
                                >
                                    <option value="">All Devices</option>
                                    <option value="h100">H100</option>
                                    <option value="a100">A100</option>
                                    <option value="v100">V100</option>
                                    <option value="rtx4090">RTX 4090</option>
                                    <option value="rtx3080">RTX 3080</option>
                                </select>
                            </div>
                            <div className="filter-group">
                                <label>Sort By</label>
                                <select 
                                    value={filters.sort} 
                                    onChange={(e) => handleFilterChange('sort', e.target.value)}
                                >
                                    <option value="performance">Performance (TFLOPS)</option>
                                    <option value="date">Submission Date</option>
                                    <option value="correctness">Correctness Score</option>
                                    <option value="author">Author</option>
                                </select>
                            </div>
                        </div>
                        <div className="view-toggle">
                            <button 
                                className={`view-btn ${currentView === 'table' ? 'active' : ''}`}
                                onClick={() => toggleView('table')}
                            >
                                <i className="fas fa-table"></i> Table View
                            </button>
                            <button 
                                className={`view-btn ${currentView === 'cards' ? 'active' : ''}`}
                                onClick={() => toggleView('cards')}
                            >
                                <i className="fas fa-th-large"></i> Card View
                            </button>
                        </div>
                    </div>

                    {/* Table View */}
                    {currentView === 'table' ? (
                        <LeaderboardTable submissions={filteredSubmissions} />
                    ) : (
                        <div className="cards-container">
                            {filteredSubmissions.map((submission, index) => {
                                const rank = index + 1;
                                return (
                                    <div key={submission.id} className={`submission-card ${rank <= 3 ? `rank-${rank}` : ''}`}>
                                        <div className="card-header">
                                            <div className="rank-badge">
                                                {rank === 1 && <i className="fas fa-crown"></i>}
                                                <span>{rank}</span>
                                            </div>
                                            <div className="submission-meta">
                                                <h4>{submission.name}</h4>
                                                <span className="submission-id">#{submission.id}</span>
                                            </div>
                                            <div className="performance-big">
                                                <span className="value">{submission.performance.toLocaleString()}</span>
                                                <span className="unit">TFLOPS</span>
                                            </div>
                                        </div>
                                        <div className="card-body">
                                            <div className="tags">
                                                <span className={`dsl-badge ${submission.dsl.toLowerCase()}`}>
                                                    {submission.dsl}
                                                </span>
                                                <span className={`device-badge ${submission.device.toLowerCase().replace(' ', '')}`}>
                                                    {submission.device}
                                                </span>
                                            </div>
                                            <div className="correctness-bar">
                                                <div className="correctness-label">Correctness: {submission.correctness}%</div>
                                                <div className="progress-bar">
                                                    <div 
                                                        className="progress-fill" 
                                                        style={{ width: `${submission.correctness}%` }}
                                                    ></div>
                                                </div>
                                            </div>
                                            <div className="author-date">
                                                <div className="author">
                                                    <img 
                                                        src="https://via.placeholder.com/24" 
                                                        alt="Author" 
                                                        className="author-avatar"
                                                    />
                                                    <span>@{submission.author}</span>
                                                </div>
                                                <span className="date">{submission.date}</span>
                                            </div>
                                        </div>
                                        <div className="card-actions">
                                            <button 
                                                className="action-btn view-btn"
                                                onClick={() => navigate(`/operator/${operatorKey}/submission/${submission.id}`)}
                                            >
                                                <i className="fas fa-eye"></i> View Details
                                            </button>
                                            <button 
                                                className="action-btn download-btn"
                                                onClick={() => alert(`Downloading ${submission.name} kernel implementation...`)}
                                            >
                                                <i className="fas fa-download"></i> Download
                                            </button>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}

                    <div className="pagination">
                        <button className="pagination-btn" disabled>
                            <i className="fas fa-chevron-left"></i> Previous
                        </button>
                        <div className="pagination-numbers">
                            <button className="pagination-number active">1</button>
                            <button className="pagination-number">2</button>
                            <button className="pagination-number">3</button>
                            <span className="pagination-dots">...</span>
                            <button className="pagination-number">12</button>
                        </div>
                        <button className="pagination-btn">
                            Next <i className="fas fa-chevron-right"></i>
                        </button>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default OperatorPage;