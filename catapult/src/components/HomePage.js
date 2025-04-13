import React from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css';

function HomePage() {
  const features = [
    {
      title: 'Advanced Detection',
      items: [
        'Detect people fainting',
        'Detect people clustered together',
        'Detect people running',
        'Identify dangerous objects'
      ]
    },
    {
      title: 'Smart Search',
      items: [
        'Search across all footage',
        'Add text annotations',
        'Add image markers',
        'Real-time alerts'
      ]
    }
  ];

  return (
    <div className="home-page">
      <header className="hero-section">
        <h1>Spectra</h1>
        <p className="tagline">Advanced Video Analytics for Enhanced Security</p>
        <Link to="/videos" className="cta-button">Try Now</Link>
      </header>

      <section className="features-section">
        <h2>Powerful Features</h2>
        <div className="features-grid">
          {features.map((feature, index) => (
            <div key={index} className="feature-card">
              <h3>{feature.title}</h3>
              <ul>
                {feature.items.map((item, itemIndex) => (
                  <li key={itemIndex}>{item}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>

      <section className="benefits-section">
        <h2>Why Choose Spectra?</h2>
        <div className="benefits-grid">
          <div className="benefit-card">
            <h3>Real-time Monitoring</h3>
            <p>Get instant alerts for critical events as they happen</p>
          </div>
          <div className="benefit-card">
            <h3>Intelligent Analysis</h3>
            <p>Advanced AI-powered detection for comprehensive security</p>
          </div>
          <div className="benefit-card">
            <h3>Easy Integration</h3>
            <p>Seamlessly works with your existing security infrastructure</p>
          </div>
        </div>
      </section>
    </div>
  );
}

export default HomePage; 