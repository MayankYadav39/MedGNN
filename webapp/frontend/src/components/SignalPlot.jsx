import React from 'react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    Filler
);

const SignalPlot = ({ data, attribution, label, color = 'var(--accent-blue)' }) => {
    // Normalize attribution for background coloring
    const maxAttr = Math.max(...(attribution || [0]));
    const minAttr = Math.min(...(attribution || [1]));

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(255, 255, 255, 0.98)',
                titleColor: '#0ea5e9',
                bodyColor: '#0f172a',
                borderColor: 'rgba(0,0,0,0.1)',
                borderWidth: 1,
            }
        },
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: 'Timestamp',
                    color: 'var(--text-secondary)',
                    font: { size: 10, weight: 600 }
                },
                grid: { display: false },
                ticks: { display: true, color: 'var(--text-secondary)', font: { size: 10 } }
            },
            y: {
                grid: { color: 'rgba(0,0,0,0.05)' },
                ticks: { color: 'var(--text-secondary)', font: { size: 10 } }
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    };

    const chartData = {
        labels: Array.from({ length: (data || []).length }, (_, i) => i),
        datasets: [
            {
                label: label,
                data: data || [],
                borderColor: color,
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.4,
            },
            {
                label: 'Importance',
                data: (data || []).map((v, i) => v), // Use same signal data for area highlighting
                backgroundColor: (context) => {
                    if (!attribution || !context.chart) return 'transparent';
                    const index = context.dataIndex;
                    const val = attribution[index];
                    const alpha = ((val - minAttr) / (maxAttr - minAttr + 1e-8)) * 0.5;
                    return `rgba(245, 158, 11, ${alpha})`; // Pulse Orange for importance
                },
                fill: true,
                pointRadius: 0,
                borderWidth: 0,
            }
        ]
    };

    return (
        <div className="glass-card" style={{ height: '200px', padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)' }}>{label}</span>
            </div>
            <div style={{ height: 'calc(100% - 24px)' }}>
                <Line options={options} data={chartData} />
            </div>
        </div>
    );
};

export default SignalPlot;
