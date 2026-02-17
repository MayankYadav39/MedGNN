import React from 'react';
import { Line } from 'react-chartjs-2';

const UncertaintyTimeline = ({ epistemic, aleatoric }) => {
    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: { boxWidth: 10, color: 'var(--text-secondary)', font: { size: 10 } }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(255, 255, 255, 0.98)',
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
                ticks: { color: 'var(--text-secondary)', font: { size: 10 } },
                beginAtZero: true
            }
        },
        interaction: {
            mode: 'nearest',
            axis: 'x',
            intersect: false
        }
    };

    const chartData = {
        labels: Array.from({ length: (epistemic || []).length }, (_, i) => i),
        datasets: [
            {
                label: 'Model Epistemic (Parameter Uncertainty)',
                data: epistemic || [],
                borderColor: 'var(--accent-blue)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.4,
            },
            {
                label: 'Data Aleatoric (Signal Noise)',
                data: aleatoric || [],
                borderColor: 'var(--accent-orange)',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                fill: true,
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.4,
            }
        ]
    };

    return (
        <div className="glass-card" style={{ height: '240px', padding: '16px' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '1rem', color: 'var(--text-secondary)' }}>Uncertainty Evolution (Temporal)</h3>
            <div style={{ height: 'calc(100% - 32px)' }}>
                <Line options={options} data={chartData} />
            </div>
        </div>
    );
};

export default UncertaintyTimeline;
