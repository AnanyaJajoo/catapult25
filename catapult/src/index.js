import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from 'react-oidc-context';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

const cognitoAuthConfig = {
  authority: `https://cognito-idp.${process.env.REACT_APP_REGION}.amazonaws.com/${process.env.REACT_APP_USER_POOL_ID}`,
  client_id: process.env.REACT_APP_USER_POOL_WEB_CLIENT_ID,
  redirect_uri: process.env.REACT_APP_REDIRECT_URI,
  response_type: "code",
  scope: "phone openid email profile",
  // Add these options for better handling
  automaticSilentRenew: true,
  loadUserInfo: true,
  onSigninCallback: (user) => {
    // After successful sign-in, clean up the URL and stay on the page
    window.history.replaceState({}, document.title, '/');
  },
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <AuthProvider {...cognitoAuthConfig}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </AuthProvider>
  </React.StrictMode>
);

reportWebVitals();