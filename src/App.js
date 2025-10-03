import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import OperatorPage from './pages/OperatorPage';
import SubmitPage from './pages/SubmitPage';
import SubmissionDetailPage from './pages/SubmissionDetailPage';
import ModelEvaluationPage from './pages/ModelEvaluationPage';
import TrainingResultPage from './pages/TrainingResultPage';
import './styles.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Header />
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/operator/:operatorKey" element={<OperatorPage />} />
            <Route path="/operator/:operatorKey/submission/:submissionId" element={<SubmissionDetailPage />} />
            <Route path="/submit" element={<SubmitPage />} />
            <Route path="/model-evaluation" element={<ModelEvaluationPage />} />
            <Route path="/model-evaluation/:modelKey/training/:submissionId" element={<TrainingResultPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </Router>
  );
}

export default App;