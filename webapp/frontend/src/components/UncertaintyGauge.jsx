import React from 'react';
import { motion } from 'framer-motion';

const UncertaintyGauge = ({ value, label, color = 'var(--accent-blue)', subLabel }) => {
    // Value is now expected to be 0-100 (percentage)
    const percentage = Math.min(100, Math.max(0, value));
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (percentage / 100) * circumference;

    return (
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flex: 1 }}>
            <h3 style={{ margin: '0 0 16px 0', fontSize: '1rem', color: 'var(--text-secondary)' }}>{label}</h3>
            <div style={{ position: 'relative', width: '120px', height: '120px' }}>
                <svg width="120" height="120" viewBox="0 0 100 100">
                    <circle
                        cx="50" cy="50" r={radius}
                        fill="transparent"
                        stroke="rgba(0,0,0,0.08)"
                        strokeWidth="8"
                    />
                    <motion.circle
                        cx="50" cy="50" r={radius}
                        fill="transparent"
                        stroke={color}
                        strokeWidth="8"
                        strokeDasharray={circumference}
                        initial={{ strokeDashoffset: circumference }}
                        animate={{ strokeDashoffset: offset }}
                        transition={{ duration: 1.5, ease: 'easeOut' }}
                        strokeLinecap="round"
                        transform="rotate(-90 50 50)"
                    />
                </svg>
                <div style={{
                    position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                    display: 'flex', flexDirection: 'column', alignItems: 'center'
                }}>
                    <span style={{ fontSize: '1.5rem', fontWeight: 700 }}>{percentage.toFixed(2)}%</span>
                </div>
            </div>
            {subLabel && <p style={{ margin: '16px 0 0 0', fontSize: '0.8rem', opacity: 0.7 }}>{subLabel}</p>}
        </div>
    );
};

export default UncertaintyGauge;
