import React, { useState, useEffect } from 'react';
import './ToastNotification.css'; // We'll create the styles for this component

const ToastNotification = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 5000); // Close after 5 seconds

    return () => clearTimeout(timer);
  }, [message, onClose]);

  return (
    <div className={`toast-notification ${type}`}>
      <span>{message}</span>
      <button className="close-btn" onClick={onClose}>×</button>
    </div>
  );
};

export default ToastNotification;
