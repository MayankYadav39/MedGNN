import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
);

const ProbabilityChart = ({ probabilities, classes }) => {
    const options = {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(16, 24, 39, 0.9)',
                titleColor: '#3b82f6',
                bodyColor: '#fff',
                borderColor: 'rgba(255,255,255,0.1)',
                borderWidth: 1,
            }
        },
        scales: {
            x: {
                grid: { color: 'rgba(255,255,255,0.05)' },
                ticks: { color: 'var(--text-secondary)', callback: (v) => `${(v * 100).toFixed(0)}%` },
                max: 1.0,
                min: 0
            },
            y: {
                grid: { display: false },
                ticks: { color: 'var(--text-primary)', font: { size: 12, weight: '600' } }
            }
        }
    };

    const data = {
        labels: classes || ['Normal', 'Risk', 'Arrhythmia', 'Critical'],
        datasets: [
            {
                data: probabilities || [0, 0, 0, 0],
                backgroundColor: (context) => {
                    const val = context.raw;
                    if (val > 0.5) return 'var(--accent-blue)';
                    if (val > 0.2) return 'rgba(59, 130, 246, 0.5)';
                    return 'rgba(255, 255, 255, 0.1)';
                },
                borderRadius: 8,
                barThickness: 32,
            }
        ]
    };

    return (
        <div className="glass-card" style={{ height: '300px', flex: 1 }}>
            <h3 style={{ margin: '0 0 20px 0', fontSize: '1.1rem', color: 'var(--text-secondary)' }}>Class Probabilities</h3>
            <div style={{ height: 'calc(100% - 40px)' }}>
                <Bar options={options} data={data} />
            </div>
        </div>
    );
};

export default ProbabilityChart;
