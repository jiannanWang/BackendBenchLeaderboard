import React from 'react';
import { Link } from 'react-router-dom';

const OperatorCard = ({ operatorKey, info, stats }) => {
    return (
        <Link 
            to={`/operator/${operatorKey}`} 
            className="operator-card" 
            style={{ textDecoration: 'none', color: 'inherit' }}
        >
            <div className="operator-icon">
                <i className={info.icon}></i>
            </div>
            <div className="operator-info">
                <h4>{info.title}</h4>
                <p>{info.description}</p>
                <div className="operator-stats">
                    <span className="stat">
                        <i className="fas fa-trophy"></i>
                        <span>{stats.submissionCount} submissions</span>
                    </span>
                    <span className="stat">
                        <i className="fas fa-fire"></i>
                        <span>{stats.topPerformance.toLocaleString()} TFLOPS</span>
                    </span>
                </div>
            </div>
            <div className="operator-arrow">
                <i className="fas fa-chevron-right"></i>
            </div>
        </Link>
    );
};

export default OperatorCard;