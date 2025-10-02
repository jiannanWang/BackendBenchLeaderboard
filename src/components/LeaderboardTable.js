import React, { useState } from 'react';
import SubmissionModal from './SubmissionModal';

const LeaderboardTable = ({ submissions }) => {
    const [selectedSubmission, setSelectedSubmission] = useState(null);

    const handleViewDetails = (submission) => {
        setSelectedSubmission(submission);
    };

    const handleDownload = (submission) => {
        alert(`Downloading ${submission.name} kernel implementation...`);
    };

    const closeModal = () => {
        setSelectedSubmission(null);
    };

    return (
        <>
            <div className="table-container">
                <table className="leaderboard-table">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Submission</th>
                            <th>DSL</th>
                            <th>Device</th>
                            <th>Performance</th>
                            <th>Correctness</th>
                            <th>Author</th>
                            <th>Date</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {submissions.map((submission, index) => {
                            const rank = index + 1;
                            return (
                                <tr key={submission.id} className={rank <= 3 ? `rank-${rank}` : ''}>
                                    <td>
                                        <div className="rank">
                                            {rank <= 3 && <i className="fas fa-crown"></i>}
                                            <span>{rank}</span>
                                        </div>
                                    </td>
                                    <td>
                                        <div className="submission-info">
                                            <span className="submission-name">{submission.name}</span>
                                            <span className="submission-id">#{submission.id}</span>
                                        </div>
                                    </td>
                                    <td>
                                        <span className={`dsl-badge ${submission.dsl.toLowerCase()}`}>
                                            {submission.dsl}
                                        </span>
                                    </td>
                                    <td>
                                        <span className={`device-badge ${submission.device.toLowerCase().replace(' ', '')}`}>
                                            {submission.device}
                                        </span>
                                    </td>
                                    <td>
                                        <div className="performance">
                                            <span className="value">{submission.performance.toLocaleString()}</span>
                                            <span className="unit">TFLOPS</span>
                                        </div>
                                    </td>
                                    <td>
                                        <div className="correctness">
                                            <span className="score">{submission.correctness}%</span>
                                            <i className="fas fa-check-circle"></i>
                                        </div>
                                    </td>
                                    <td>
                                        <div className="author">
                                            <img 
                                                src="https://via.placeholder.com/32" 
                                                alt="Author" 
                                                className="author-avatar"
                                            />
                                            <span>@{submission.author}</span>
                                        </div>
                                    </td>
                                    <td>
                                        <span className="date">{submission.date}</span>
                                    </td>
                                    <td>
                                        <div className="actions">
                                            <button 
                                                className="action-btn view-btn" 
                                                title="View Details"
                                                onClick={() => handleViewDetails(submission)}
                                            >
                                                <i className="fas fa-eye"></i>
                                            </button>
                                            <button 
                                                className="action-btn download-btn" 
                                                title="Download"
                                                onClick={() => handleDownload(submission)}
                                            >
                                                <i className="fas fa-download"></i>
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {selectedSubmission && (
                <SubmissionModal 
                    submission={selectedSubmission} 
                    onClose={closeModal} 
                />
            )}
        </>
    );
};

export default LeaderboardTable;