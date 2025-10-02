import React from 'react';

const Footer = () => {
    return (
        <footer className="footer">
            <div className="container">
                <div className="footer-content">
                    <div className="footer-section">
                        <h4>BackendBench</h4>
                        <p>Open source GPU kernel benchmarking platform for researchers and developers.</p>
                    </div>
                    <div className="footer-section">
                        <h4>Resources</h4>
                        <ul>
                            <li><a href="#docs">Documentation</a></li>
                            <li><a href="#api">API Reference</a></li>
                            <li><a href="#examples">Examples</a></li>
                            <li><a href="#faq">FAQ</a></li>
                        </ul>
                    </div>
                    <div className="footer-section">
                        <h4>Community</h4>
                        <ul>
                            <li><a href="#github">GitHub</a></li>
                            <li><a href="#discord">Discord</a></li>
                            <li><a href="#twitter">Twitter</a></li>
                            <li><a href="#blog">Blog</a></li>
                        </ul>
                    </div>
                    <div className="footer-section">
                        <h4>Support</h4>
                        <ul>
                            <li><a href="#contact">Contact</a></li>
                            <li><a href="#issues">Report Issues</a></li>
                            <li><a href="#feedback">Feedback</a></li>
                        </ul>
                    </div>
                </div>
                <div className="footer-bottom">
                    <p>&copy; 2024 BackendBench. Open source project under MIT License.</p>
                </div>
            </div>
        </footer>
    );
};

export default Footer;