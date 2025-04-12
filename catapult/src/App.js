import React, { useState, useRef, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate, useNavigate } from 'react-router-dom';
import { withAuthenticator } from '@aws-amplify/ui-react';
import { signOut, getCurrentUser } from '@aws-amplify/auth';
import './App.css';
import './aws-config';
import AuthComponent from './components/Auth';

const API_URL = 'http://localhost:9000';

function NavBar({ isAuthenticated, onSignOut }) {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="nav-links">
        <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
          All Videos
        </Link>
        <Link to="/favorites" className={location.pathname === '/favorites' ? 'active' : ''}>
          Favorites
        </Link>
      </div>
      {isAuthenticated ? (
        <button onClick={onSignOut} className="sign-out-button">
          Sign Out
        </button>
      ) : (
        <Link to="/auth" className="auth-link">
          Sign In
        </Link>
      )}
    </nav>
  );
}

function VideoGrid({ videos, onVideoClick, onToggleFavorite, showFavoritesOnly = false }) {
  const displayedVideos = showFavoritesOnly 
    ? videos.filter(video => video.isFavorite)
    : videos;

  return (
    <div className="video-grid">
      {displayedVideos.map((video, index) => {
        console.log('Rendering video:', video);
        return (
          <div 
            key={index} 
            className="video-thumbnail"
            onClick={() => onVideoClick(video, index)}
          >
            <iframe 
              src={video.url}
              className="thumbnail-preview"
              title={video.name}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              onError={(e) => {
                console.error('Error loading video:', video.url, e);
                e.target.style.display = 'none';
              }}
            />
            <div className="video-name">{video.location || video.name}</div>
            <button 
              className={`favorite-button ${video.isFavorite ? 'favorited' : ''}`}
              onClick={(e) => {
                e.stopPropagation();
                onToggleFavorite(index);
              }}
            >
              ★
            </button>
          </div>
        );
      })}
    </div>
  );
}

function LocationPopup({ isOpen, onClose, onSave, videoName }) {
  const [location, setLocation] = useState('');
  const [customName, setCustomName] = useState('');

  if (!isOpen) return null;

  return (
    <div className="popup-overlay">
      <div className="popup-content">
        <h2>Add Details for {videoName}</h2>
        <input
          type="text"
          placeholder="Enter video name"
          value={customName}
          onChange={(e) => setCustomName(e.target.value)}
          className="location-input"
        />
        <input
          type="text"
          placeholder="Enter video location"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          className="location-input"
        />
        <div className="popup-buttons">
          <button className="popup-button cancel" onClick={onClose}>Cancel</button>
          <button className="popup-button save" onClick={() => {
            onSave(location, customName);
            setLocation('');
            setCustomName('');
          }}>Save</button>
        </div>
      </div>
    </div>
  );
}

function VideoPlayer({ video, onClose }) {
  if (!video) return null;

  return (
    <div className="video-player-overlay">
      <div className="video-player-container">
        <button className="close-button" onClick={onClose}>×</button>
        <div className="video-player">
          <iframe
            src={video.url}
            title={video.name}
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
            className="video-frame"
          />
        </div>
        <div className="video-info">
          <h2>{video.location || video.name}</h2>
        </div>
      </div>
    </div>
  );
}

function VideoPage({ videos, onVideoClick, onToggleFavorite, showFavoritesOnly }) {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [showLocationPopup, setShowLocationPopup] = useState(false);
  const [pendingVideo, setPendingVideo] = useState(null);
  const fileInputRef = useRef(null);
  const location = useLocation();

  const handleVideoClick = (video, index) => {
    setSelectedVideo(video);
  };

  const handleClosePlayer = () => {
    setSelectedVideo(null);
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      const videoUrl = URL.createObjectURL(file);
      setPendingVideo({ url: videoUrl, name: file.name });
      setShowLocationPopup(true);
    }
  };

  const handleLocationSave = (location, customName) => {
    if (pendingVideo) {
      onVideoClick({ 
        ...pendingVideo, 
        location, 
        customName,
        isFavorite: false 
      }, videos.length);
      setPendingVideo(null);
    }
    setShowLocationPopup(false);
  };

  return (
    <div className="video-page">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
        style={{ display: 'none' }}
      />
      {!showFavoritesOnly && (
        <button className="upload-button" onClick={handleUploadClick}>Upload Video</button>
      )}
      
      <VideoGrid
        videos={videos}
        onVideoClick={handleVideoClick}
        onToggleFavorite={onToggleFavorite}
        showFavoritesOnly={showFavoritesOnly}
      />

      <LocationPopup
        isOpen={showLocationPopup}
        onClose={() => setShowLocationPopup(false)}
        onSave={handleLocationSave}
        videoName={pendingVideo?.name || ''}
      />

      {selectedVideo && (
        <VideoPlayer
          video={selectedVideo}
          onClose={handleClosePlayer}
        />
      )}
    </div>
  );
}

function App() {
  const [videos, setVideos] = useState([]);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);

  const checkAuth = async () => {
    try {
      const user = await getCurrentUser();
      setIsAuthenticated(true);
      setCurrentUser(user);
      fetchUserVideos(user.username);
    } catch (error) {
      setIsAuthenticated(false);
      setCurrentUser(null);
    }
  };

  useEffect(() => {
    checkAuth();
  }, []);

  const fetchUserVideos = async (username) => {
    try {
      const response = await fetch(`${API_URL}/videos?user=${username}`, {
        credentials: 'include'
      });
      if (!response.ok) {
        throw new Error('Failed to fetch videos');
      }
      const data = await response.json();
      setVideos(data.map(video => ({
        ...video,
        isFavorite: false,
      })));
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

  const handleSignOut = async () => {
    try {
      await signOut();
      setIsAuthenticated(false);
      setCurrentUser(null);
      setVideos([]);
    } catch (error) {
      console.error('Error signing out:', error);
    }
  };

  const handleAuthStateChange = (authenticated) => {
    setIsAuthenticated(authenticated);
    if (authenticated) {
      checkAuth();
    }
  };

  const toggleFavorite = (videoIndex) => {
    setVideos(prevVideos => {
      const newVideos = [...prevVideos];
      newVideos[videoIndex] = {
        ...newVideos[videoIndex],
        isFavorite: !newVideos[videoIndex].isFavorite
      };
      return newVideos;
    });
  };

  return (
    <div className="app">
      <NavBar isAuthenticated={isAuthenticated} onSignOut={handleSignOut} />
      <Routes>
        <Route
          path="/auth"
          element={
            isAuthenticated ? (
              <Navigate to="/" replace />
            ) : (
              <AuthComponent onAuthStateChange={handleAuthStateChange} />
            )
          }
        />
        <Route
          path="/"
          element={
            isAuthenticated ? (
              <VideoPage
                videos={videos}
                onVideoClick={toggleFavorite}
                onToggleFavorite={toggleFavorite}
                showFavoritesOnly={false}
              />
            ) : (
              <Navigate to="/auth" replace />
            )
          }
        />
        <Route
          path="/favorites"
          element={
            isAuthenticated ? (
              <VideoPage
                videos={videos}
                onVideoClick={toggleFavorite}
                onToggleFavorite={toggleFavorite}
                showFavoritesOnly={true}
              />
            ) : (
              <Navigate to="/auth" replace />
            )
          }
        />
      </Routes>
    </div>
  );
}

// Wrap the App component with withAuthenticator
const AppWithAuth = withAuthenticator(App);

// Create a wrapper component to handle routing
const AppWrapper = () => {
  return (
    <Router>
      <AppWithAuth />
    </Router>
  );
};

export default AppWrapper;
