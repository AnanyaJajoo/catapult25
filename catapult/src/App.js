import React, { useState, useRef, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { s3 } from './aws-config';
import HomePage from './components/HomePage';
import './App.css';

function NavBar() {
  const [searchQuery, setSearchQuery] = useState('');
  const location = useLocation();
  const isHomePage = location.pathname === '/';

  if (isHomePage) return null;

  return (
    <nav className="nav-bar">
      <Link to="/" className="logo">Spectra</Link>
      <div className="nav-links">
        <Link to="/">Home</Link>
        <Link to="/videos">All Videos</Link>
        <Link to="/favorites">Favorites</Link>
        <div className="search-container">
          <input
            type="text"
            placeholder="Search videos..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="search-input"
          />
        </div>
      </div>
    </nav>
  );
}

function VideoGrid({ videos, onVideoClick, onToggleFavorite, showFavoritesOnly = false, isPlaying = false }) {
  const displayedVideos = showFavoritesOnly 
    ? videos.filter(video => video.isFavorite)
    : videos;

  return (
    <div className="video-grid">
      {displayedVideos.map((video, index) => (
        <div 
          key={index} 
          className="video-thumbnail"
          onClick={() => onVideoClick(video, index)}
        >
          <video 
            src={video.url}
            className="thumbnail-preview"
            title={video.customName || video.name}
            muted
            autoPlay
            loop
            playsInline
            style={{ pointerEvents: 'none' }}
          />
          <div className="video-info-container">
            <div className="video-name">{video.customName || video.name}</div>
            <div className="video-location">{video.location || 'No location specified'}</div>
          </div>
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
      ))}
    </div>
  );
}

function LocationPopup({ isOpen, onClose, onSave, videoName }) {
  const [location, setLocation] = useState('');
  const [customName, setCustomName] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (isOpen) {
      setLocation('');
      setCustomName('');
      setError('');
    }
  }, [isOpen]);

  const handleSave = () => {
    if (!customName.trim()) {
      setError('Please enter a video name');
      return;
    }
    onSave(location.trim(), customName.trim());
  };

  if (!isOpen) return null;

  return (
    <div className="popup-overlay">
      <div className="popup-content">
        <h2>Add Video Details</h2>
        <div className="popup-form">
          <div className="form-group">
            <label htmlFor="video-name">Video Name</label>
            <input
              id="video-name"
              type="text"
              placeholder="Enter video name"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
              className="location-input"
            />
            {error && <span className="error-message">{error}</span>}
          </div>
          <div className="form-group">
            <label htmlFor="video-location">Location</label>
            <input
              id="video-location"
              type="text"
              placeholder="Enter video location"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              className="location-input"
            />
          </div>
        </div>
        <div className="popup-buttons">
          <button className="popup-button cancel" onClick={onClose}>Cancel</button>
          <button className="popup-button save" onClick={handleSave}>Save</button>
        </div>
      </div>
    </div>
  );
}

function VideoPlayer({ video, onClose, onNext, onPrev, hasNext, hasPrev }) {
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const videoRef = useRef(null);
  const [isExpanded, setIsExpanded] = useState(false);

  // Mock incidents data - replace with your actual data
  const incidents = [
    { time: 30, description: "Person detected" },
    { time: 120, description: "Motion detected" },
    { time: 180, description: "Object detected" },
    { time: 240, description: "Person detected" },
  ];

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (videoRef.current) {
      setDuration(videoRef.current.duration);
    }
  };

  const handleTimestampClick = (time) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
    }
  };

  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  if (!video) return null;

  return (
    <div className={`video-player-overlay ${isExpanded ? 'expanded' : ''}`}>
      <div className="video-player-container">
        <button className="close-button" onClick={onClose}>×</button>
        <div className="video-player-content">
          <div className="video-player-main">
            <div className="video-player">
              <video
                ref={videoRef}
                src={video.url}
                title={video.name}
                controls
                autoPlay
                className="video-frame"
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
              />
            </div>
            <div className="video-info">
              <h2>{video.location || video.name}</h2>
              <div className="video-time">
                {formatTime(currentTime)} / {formatTime(duration)}
              </div>
            </div>
            <div className="video-navigation">
              <button 
                className="nav-button prev" 
                onClick={onPrev}
                disabled={!hasPrev}
              >
                ← Previous
              </button>
              <button 
                className="nav-button next" 
                onClick={onNext}
                disabled={!hasNext}
              >
                Next →
              </button>
            </div>
          </div>
          <div className="video-timeline">
            <h3>Timeline</h3>
            <div className="incidents-list">
              {incidents.map((incident, index) => (
                <div 
                  key={index}
                  className={`incident-item ${currentTime >= incident.time && currentTime < incident.time + 5 ? 'active' : ''}`}
                  onClick={() => handleTimestampClick(incident.time)}
                >
                  <div className="incident-time">{formatTime(incident.time)}</div>
                  <div className="incident-description">{incident.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function VideoPage({ videos, onVideoClick, onToggleFavorite, showFavoritesOnly = false, searchQuery = '' }) {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [showLocationPopup, setShowLocationPopup] = useState(false);
  const [pendingVideo, setPendingVideo] = useState(null);
  const fileInputRef = useRef(null);

  // Filter videos based on search query
  const filteredVideos = videos.filter(video => {
    if (!searchQuery) return true;
    const searchLower = searchQuery.toLowerCase();
    return (
      (video.customName && video.customName.toLowerCase().includes(searchLower)) ||
      (video.location && video.location.toLowerCase().includes(searchLower)) ||
      (video.name && video.name.toLowerCase().includes(searchLower))
    );
  });

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith('video/')) {
      try {
        const key = `videos/${Date.now()}-${file.name}`;
        
        await s3.putObject({
          Bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
          Key: key,
          Body: file,
          ContentType: file.type
        }).promise();
        
        const url = await s3.getSignedUrlPromise('getObject', {
          Bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
          Key: key,
          Expires: 3600
        });
        
        const newVideo = {
          url,
          name: file.name,
          key: key,
          isFavorite: false
        };
        
        setPendingVideo(newVideo);
        setShowLocationPopup(true);
      } catch (error) {
        console.error('Error uploading video:', error);
      }
    }
  };

  const handleLocationSave = (location, customName) => {
    if (pendingVideo) {
      const updatedVideo = {
        ...pendingVideo,
        location,
        customName
      };
      onVideoClick(updatedVideo);
      setPendingVideo(null);
    }
    setShowLocationPopup(false);
  };

  const handleVideoClick = (video, index) => {
    setSelectedVideo({ ...video, index });
  };

  const handleCloseVideo = () => {
    setSelectedVideo(null);
  };

  const handleNextVideo = () => {
    if (selectedVideo && selectedVideo.index < filteredVideos.length - 1) {
      setSelectedVideo({ ...filteredVideos[selectedVideo.index + 1], index: selectedVideo.index + 1 });
    }
  };

  const handlePrevVideo = () => {
    if (selectedVideo && selectedVideo.index > 0) {
      setSelectedVideo({ ...filteredVideos[selectedVideo.index - 1], index: selectedVideo.index - 1 });
    }
  };

  return (
    <div className="App">
      <NavBar onSignOut={handleSignOut} />
      <div className="user-info">
        <p>Signed in as: {auth.user?.profile.email || auth.user?.profile.sub}</p>
      </div>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="video/*"
        style={{ display: 'none' }}
      />
      {!showFavoritesOnly && (
        <div className="upload-button-container">
          <button className="upload-button" onClick={handleUploadClick}>Upload Video</button>
        </div>
      )}
      
      <VideoGrid
        videos={filteredVideos}
        onVideoClick={handleVideoClick}
        onToggleFavorite={onToggleFavorite}
        showFavoritesOnly={showFavoritesOnly}
        isPlaying={!selectedVideo}
      />

      {selectedVideo && (
        <VideoPlayer
          video={selectedVideo}
          onClose={handleCloseVideo}
          onNext={handleNextVideo}
          onPrev={handlePrevVideo}
          hasNext={selectedVideo.index < filteredVideos.length - 1}
          hasPrev={selectedVideo.index > 0}
        />
      )}

      <LocationPopup
        isOpen={showLocationPopup}
        onClose={() => setShowLocationPopup(false)}
        onSave={handleLocationSave}
        videoName={pendingVideo?.name || ''}
      />
    </div>
  );
}

function App() {
  const [videos, setVideos] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  const fetchUserVideos = async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const params = {
        Bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
        Prefix: 'videos/'
      };

      console.log('Fetching videos with params:', params);
      const data = await s3.listObjectsV2(params).promise();
      console.log('Received S3 data:', data);
      
      const videos = await Promise.all(
        data.Contents.map(async (item) => {
          const url = await s3.getSignedUrlPromise('getObject', {
            Bucket: process.env.REACT_APP_AWS_BUCKET_NAME,
            Key: item.Key,
            Expires: 3600
          });
          
          return {
            url,
            name: item.Key.split('/').pop(),
            key: item.Key,
            isFavorite: false
          };
        })
      );

      console.log('Processed videos:', videos);
      setVideos(videos);
    } catch (error) {
      console.error('Error fetching videos:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchUserVideos();
  }, []);

  const handleVideoClick = (video) => {
    setVideos(prevVideos => [...prevVideos, video]);
  };

  const handleToggleFavorite = (index) => {
    const updatedVideos = [...videos];
    updatedVideos[index].isFavorite = !updatedVideos[index].isFavorite;
    setVideos(updatedVideos);
  };

  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading videos...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error Loading Videos</h2>
        <p>{error}</p>
        <button onClick={fetchUserVideos}>Try Again</button>
      </div>
    );
  }

  return (
    <div className="main-wrapper">
      <NavBar searchQuery={searchQuery} onSearchChange={setSearchQuery} />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/videos" element={
          <VideoPage
            videos={videos}
            onVideoClick={handleVideoClick}
            onToggleFavorite={handleToggleFavorite}
            showFavoritesOnly={false}
            searchQuery={searchQuery}
          />
        } />
        <Route path="/favorites" element={
          <VideoPage
            videos={videos}
            onVideoClick={handleVideoClick}
            onToggleFavorite={handleToggleFavorite}
            showFavoritesOnly={true}
            searchQuery={searchQuery}
          />
        } />
      </Routes>
    </div>
  );
}

function VideoGrid({ videos, onToggleFavorite }) {
  return (
    <div className="video-grid">
      {videos.length === 0 ? (
        <h3>No videos found</h3>
      ) : (
        videos.map((video, index) => (
          <div key={index} className="video-thumbnail">
            <video
              src={video.url}
              className="thumbnail-preview"
              controls
            />
            <div className="video-info">
              <span className="video-name">{video.name}</span>
              <button 
                className={`favorite-button ${video.isFavorite ? 'favorited' : ''}`}
                onClick={() => onToggleFavorite(index)}
              >
                ★
              </button>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

export default App;