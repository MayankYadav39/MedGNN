import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, Brain, ShieldCheck, AlertTriangle, FileText, ChevronRight, Settings, User } from 'lucide-react';
import UncertaintyGauge from './components/UncertaintyGauge';
import SignalPlot from './components/SignalPlot';
import ProbabilityChart from './components/ProbabilityChart';
import UncertaintyTimeline from './components/UncertaintyTimeline';

// Configuration: Change this to your Ngrok Backend URL if tunneling
const API_BASE_URL = "http://localhost:8000";

const App = () => {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [selectedSample, setSelectedSample] = useState(0);

  // Mock data for initial load
  const loadMockData = () => {
    const T = 256;
    const C = 4;
    const mockSignal = Array.from({ length: T }, () => Array.from({ length: C }, () => Math.random() - 0.5));

    setData({
      prediction: [0],
      probabilities: [[0, 0, 0, 0]],
      uncertainty: { total: [0], epistemic: [0], aleatoric: [0] },
      signal: mockSignal
    });

    setExplanation({
      top_features_prediction: [
        { name: 'Feature 1', importance: 0 },
        { name: 'Feature 2', importance: 0 },
        { name: 'Feature 3', importance: 0 }
      ],
      top_features_uncertainty: [
        { name: 'Feature 1 (Signal)', importance: 0 },
        { name: 'Feature 2 (Noise)', importance: 0 }
      ],
      prediction_attribution: [Array.from({ length: T }, () => Array.from({ length: C }, () => 0))],
      uncertainty_attribution: [Array.from({ length: T }, () => Array.from({ length: C }, () => 0))]
    });
  };

  useEffect(() => {
    loadMockData();
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    setExplanation(null);
    try {
      // 1. Fetch specific sample
      const sampleRes = await axios.get(`${API_BASE_URL}/sample/${selectedSample}`, {
        headers: { "ngrok-skip-browser-warning": "true" }
      });
      const signal = sampleRes.data.data;

      // 2. Predict & Explain
      const [res, exp] = await Promise.all([
        axios.post(`${API_BASE_URL}/predict`, { data: [signal] }, { headers: { "ngrok-skip-browser-warning": "true" } }),
        axios.post(`${API_BASE_URL}/explain`, { data: [signal], steps: 10 }, { headers: { "ngrok-skip-browser-warning": "true" } })
      ]);

      setData({
        signal: signal,
        label: sampleRes.data.label,
        prediction: res.data.prediction,
        probabilities: res.data.probabilities,
        uncertainty: {
          total: res.data.uncertainty.total.map(v => v * 100),
          epistemic: res.data.uncertainty.epistemic.map(v => v * 100),
          aleatoric: res.data.uncertainty.aleatoric.map(v => v * 100)
        }
      });

      setExplanation(exp.data);
    } catch (err) {
      console.error("API Error:", err);
      alert("Analysis failed. Backend might be loading model or index out of range.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid-dashboard">
      {/* Sidebar */}
      <aside className="sidebar">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '10px' }}
        >
          <div className="btn-primary" style={{ padding: '10px', borderRadius: '14px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Activity size={24} />
          </div>
          <div>
            <h2 className="brand-font" style={{ margin: 0, fontSize: '1.6rem', lineHeight: 1 }}>
              X-MedBayes <span className="brand-gradient" style={{ fontSize: '0.9rem' }}>PRO</span>
            </h2>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', fontWeight: 700, letterSpacing: '0.1em' }}>EXPLAINABLE MEDICAL AI</div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card"
          style={{ padding: '24px', background: 'linear-gradient(180deg, rgba(14, 165, 233, 0.03), transparent)' }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '16px' }}>
            <div style={{ background: 'rgba(14, 165, 233, 0.1)', padding: '8px', borderRadius: '10px' }}>
              <User size={18} color="var(--accent-blue)" />
            </div>
            <div>
              <div className="stat-label">Active Subject</div>
              <div style={{ fontWeight: 700, fontSize: '1rem' }}>Patient Cluster A-1</div>
            </div>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '12px', borderTop: '1px solid var(--glass-border)' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Status</span>
            <span style={{ fontSize: '0.75rem', color: 'var(--accent-emerald)', fontWeight: 700 }}>LIVE ANALYSIS</span>
          </div>
        </motion.div>

        <nav style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
          {['Overview', 'Analysis', 'Time Series', 'XAI Report', 'System Settings'].map((item, i) => (
            <motion.div
              key={item}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 + (i * 0.05) }}
              className="glass-card"
              style={{
                padding: '14px 20px', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                background: i === 0 ? 'rgba(14, 165, 233, 0.08)' : 'var(--panel-bg)',
                borderColor: i === 0 ? 'var(--accent-blue)' : 'var(--glass-border)'
              }}
            >
              <span style={{ fontSize: '0.9rem', fontWeight: 600, color: i === 0 ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{item}</span>
              <ChevronRight size={16} color={i === 0 ? 'var(--accent-blue)' : 'var(--text-muted)'} />
            </motion.div>
          ))}
        </nav>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
            <h1 style={{ margin: 0, fontSize: '2.4rem', fontWeight: 800 }} className="glow-text">Dashboard</h1>
            <p style={{ margin: '6px 0 0 0', color: 'var(--text-muted)', fontSize: '1rem', fontWeight: 500 }}>
              Transparent Clinical Inference with Uncertainty Quantification
            </p>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{ display: 'flex', gap: '16px', alignItems: 'center' }}
          >
            <div className="glass-card" style={{ padding: '10px 20px', display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span className="stat-label">Subject Index</span>
              <input
                type="number"
                value={selectedSample}
                onChange={(e) => setSelectedSample(parseInt(e.target.value) || 0)}
                style={{
                  background: 'rgba(0,0,0,0.03)', border: '1px solid var(--glass-border)', color: 'var(--text-primary)',
                  width: '50px', outline: 'none', fontWeight: 800, padding: '4px 8px', borderRadius: '6px', textAlign: 'center'
                }}
              />
            </div>
            <button className="btn-primary" onClick={handlePredict} disabled={loading}>
              {loading ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div className="pulse-indicator" style={{ backgroundColor: 'white' }}></div>
                  <span>Processing</span>
                </div>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <Brain size={18} />
                  <span>Execute Model</span>
                </div>
              )}
            </button>
          </motion.div>
        </header>

        {/* Prediction Summary */}
        <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr 1fr', gap: '24px' }}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="glass-card"
            style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', background: 'linear-gradient(135deg, rgba(14, 165, 233, 0.08), rgba(6, 182, 212, 0.05))' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <Brain size={20} color="var(--accent-blue)" />
              <span className="stat-label">Diagnostic Logic</span>
            </div>
            <h2 style={{ fontSize: '2.8rem', margin: '4px 0', fontWeight: 900 }} className="brand-gradient">
              {data ? (data.prediction[0] === 0 ? "Alzheimer's" : "Healthy") : '---'}
            </h2>
            <div style={{ display: 'flex', gap: '8px', marginTop: '12px', alignItems: 'center' }}>
              <div className="pulse-indicator" style={{ backgroundColor: 'var(--accent-emerald)' }}></div>
              <span style={{ fontSize: '0.85rem', color: 'var(--accent-emerald)', fontWeight: 700, letterSpacing: '0.05em' }}>
                {data?.probabilities[0][data.prediction[0]] > 0.8 ? 'OPTIMAL CONFIDENCE' : 'VERIFICATION NEEDED'}
              </span>
            </div>
          </motion.div>

          <UncertaintyGauge
            label="Confidence Variance"
            value={data?.uncertainty.total[0] || 0}
            color="var(--accent-violet)"
          />
          <UncertaintyGauge
            label="Epistemic (Model)"
            value={data?.uncertainty.epistemic[0] || 0}
            color="var(--accent-blue)"
          />
          <UncertaintyGauge
            label="Aleatoric (Data)"
            value={data?.uncertainty.aleatoric[0] || 0}
            color="var(--accent-amber)"
          />
        </div>

        {/* Analytics Section */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
          <motion.div initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
            <ProbabilityChart
              probabilities={data?.probabilities[0]}
              classes={["Alzheimer's (AD)", "Healthy Control (HC)"]}
            />
          </motion.div>

          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
            style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}
          >
            <UncertaintyTimeline
              epistemic={explanation?.uncertainty_attribution?.[0]?.map(step => Math.max(...step) * 10000) || []}
              aleatoric={explanation?.uncertainty_attribution?.[0]?.map(step => Math.abs(Math.min(...step)) * 10000) || []}
            />
            <div className="glass-card" style={{ padding: '24px', display: 'flex', alignItems: 'center', gap: '20px', background: 'rgba(0,0,0,0.01)' }}>
              <div style={{ background: 'rgba(14, 165, 233, 0.1)', padding: '12px', borderRadius: '14px' }}>
                <FileText size={28} color="var(--accent-blue)" />
              </div>
              <div>
                <div className="stat-label" style={{ marginBottom: '4px' }}>AI Clinical Briefing</div>
                <p style={{ margin: 0, fontSize: '0.95rem', fontWeight: 500, lineHeight: 1.4 }}>
                  {data?.uncertainty.total[0] > 0.3
                    ? "Model flags high aleatoric noise. Secondary validation via alternative modality recommended."
                    : "Statistical stability achieved. Prediction aligns with high-resolution topographical clusters."}
                </p>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Signal Analysis */}
        <motion.section initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px', marginBottom: '24px' }}>
            <div style={{ background: 'rgba(34, 211, 238, 0.1)', padding: '10px', borderRadius: '12px' }}>
              <Activity size={20} color="var(--accent-cyan)" />
            </div>
            <h2 style={{ fontSize: '1.5rem', margin: 0, fontWeight: 700 }}>Spatial-Temporal Vitals</h2>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
            <SignalPlot
              label="Feature 1"
              data={data?.signal.map(s => s[0])}
              attribution={explanation?.prediction_attribution?.[0]?.map(s => s[0])}
              color="var(--accent-cyan)"
            />
            <SignalPlot
              label="Feature 2"
              data={data?.signal.map(s => s[1])}
              attribution={explanation?.prediction_attribution?.[0]?.map(s => s[1])}
              color="var(--accent-blue)"
            />
            <SignalPlot
              label="Feature 3"
              data={data?.signal.map(s => s[2])}
              attribution={explanation?.uncertainty_attribution?.[0]?.map(s => s[2])}
              color="var(--accent-violet)"
            />
            <SignalPlot
              label="Feature 4"
              data={data?.signal.map(s => s[3])}
              attribution={explanation?.uncertainty_attribution?.[0]?.map(s => s[3])}
              color="var(--accent-amber)"
            />
          </div>
        </motion.section>

        {/* XAI Evidence Report */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '32px' }}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="glass-card"
            style={{ borderLeft: '4px solid var(--accent-emerald)' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
              <ShieldCheck size={24} color="var(--accent-emerald)" />
              <h3 style={{ margin: 0, fontSize: '1.2rem' }}>Supporting Evidence</h3>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {explanation?.reasoning?.supporting_evidence.map((feat, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '14px 18px', borderRadius: '12px', background: 'rgba(16, 185, 129, 0.04)', border: '1px solid rgba(16, 185, 129, 0.1)' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>{feat.name}</span>
                  <span style={{ fontWeight: 800, color: 'var(--accent-emerald)', fontFamily: 'monospace' }}>+{feat.importance.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 }}
            className="glass-card"
            style={{ borderLeft: '4px solid var(--accent-rose)' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
              <AlertTriangle size={24} color="var(--accent-rose)" />
              <h3 style={{ margin: 0, fontSize: '1.2rem' }}>Contradicting Evidence</h3>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              {explanation?.reasoning?.contradicting_evidence.map((feat, i) => (
                <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: '14px 18px', borderRadius: '12px', background: 'rgba(244, 63, 92, 0.04)', border: '1px solid rgba(244, 63, 92, 0.1)' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.95rem' }}>{feat.name}</span>
                  <span style={{ fontWeight: 800, color: 'var(--accent-rose)', fontFamily: 'monospace' }}>-{feat.importance.toFixed(4)}</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Uncertainty Sources */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass-card"
          style={{ marginBottom: '40px', background: 'linear-gradient(180deg, rgba(139, 92, 246, 0.03), transparent)' }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
            <Brain size={24} color="var(--accent-violet)" />
            <h3 style={{ margin: 0, fontSize: '1.2rem' }}>Neural Entropy Contributors (Ambiguity Analysis)</h3>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
            {explanation?.top_features_uncertainty.map((feat, i) => (
              <div key={i} className="glass-card" style={{ padding: '16px', background: 'rgba(139, 92, 246, 0.02)', border: '1px solid rgba(139, 92, 246, 0.15)' }}>
                <span className="stat-label" style={{ color: 'var(--accent-violet)' }}>Channel Interference {i + 1}</span>
                <div style={{ fontWeight: 700, fontSize: '1.1rem', margin: '6px 0' }}>{feat.name}</div>
                <div style={{ width: '100%', height: '6px', background: 'rgba(139, 92, 246, 0.1)', borderRadius: '3px', marginTop: '10px', overflow: 'hidden' }}>
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.min(100, feat.importance * 1000)}%` }}
                    transition={{ duration: 1, delay: 0.8 }}
                    style={{ height: '100%', background: 'linear-gradient(90deg, var(--accent-violet), var(--accent-fuchsia))' }}
                  ></motion.div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </main>
    </div>
  );
};

export default App;
