import React from 'react';
import OperatorCard from '../components/OperatorCard';
import { operatorInfo, getOperatorStats, getGlobalStats } from '../data/mockData';

const HomePage = () => {
    const operatorStats = getOperatorStats();
    const globalStats = getGlobalStats();

    return (
        <div>
            {/* Hero Section */}
            <section className="hero">
                <div className="container">
                    <div className="hero-content">
                        <h2>Benchmark Your GPU Kernels</h2>
                        <p>Compare performance across different implementations, DSLs, and devices. Built for researchers training NanoGPT and beyond.</p>
                        <div className="stats">
                            <div className="stat">
                                <span className="stat-number">{globalStats.totalSubmissions}</span>
                                <span className="stat-label">Total Submissions</span>
                            </div>
                            <div className="stat">
                                <span className="stat-number">{globalStats.uniqueContributors}</span>
                                <span className="stat-label">Contributors</span>
                            </div>
                            <div className="stat">
                                <span className="stat-number">{globalStats.totalOperators}</span>
                                <span className="stat-label">Operators</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* Operators Grid */}
            <section className="operators" id="operators">
                <div className="container">
                    <div className="operators-header">
                        <h3>Select an Operator to View Rankings</h3>
                        <p>Click on any operator below to see the performance leaderboard for that specific kernel type</p>
                    </div>

                    <div className="operators-grid">
                        {Object.entries(operatorInfo).map(([operatorKey, info]) => (
                            <OperatorCard
                                key={operatorKey}
                                operatorKey={operatorKey}
                                info={info}
                                stats={operatorStats[operatorKey]}
                            />
                        ))}
                    </div>
                </div>
            </section>
        </div>
    );
};

export default HomePage;