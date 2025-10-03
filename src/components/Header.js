import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Header = () => {
    const location = useLocation();

    return (
        <header className="header">
            <div className="container">
                <div className="header-content">
                    <div className="logo">
                        <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
                            <h1><i className="fas fa-microchip"></i> BackendBench</h1>
                            <span className="subtitle">Kernel Performance Leaderboard</span>
                        </Link>
                    </div>
                    <nav className="nav">
                        <Link 
                            to="/" 
                            className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
                        >
                            Leaderboard
                        </Link>
                        <Link 
                            to="/model-evaluation" 
                            className={`nav-link ${location.pathname.startsWith('/model-evaluation') ? 'active' : ''}`}
                        >
                            Model Evaluation
                        </Link>
                        <Link 
                            to="/submit" 
                            className={`nav-link ${location.pathname === '/submit' ? 'active' : ''}`}
                        >
                            Submit
                        </Link>
                        <a href="#docs" className="nav-link">Documentation</a>
                        <a href="#about" className="nav-link">About</a>
                    </nav>
                    <Link to="/submit" className="submit-btn">
                        <i className="fas fa-plus"></i> Submit Kernel
                    </Link>
                </div>
            </div>
        </header>
    );
};

export default Header;