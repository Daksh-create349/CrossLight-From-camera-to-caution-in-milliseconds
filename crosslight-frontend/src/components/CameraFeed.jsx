import React, { useState, useEffect } from 'react';
import { Video, VideoOff, Play } from 'lucide-react';

export default function CameraFeed({ projectorActive, connected }) {
  const [offline, setOffline] = useState(false);
  const [feedUrl, setFeedUrl] = useState('/api/video_feed');

  // Trigger reload when connection status restores
  useEffect(() => {
    if (connected) {
      setOffline(false);
      // Append a cache buster query parameter to force reconnecting the MJPEG stream
      setFeedUrl(`/api/video_feed?t=${Date.now()}`);
    } else {
      setOffline(true);
    }
  }, [connected]);

  const handleImageError = () => {
    setOffline(true);
  };

  return (
    <div 
      className={`relative rounded-2xl overflow-hidden border transition-all duration-500 bg-gray-950/80 aspect-[16/9] flex items-center justify-center ${
        projectorActive && connected
          ? 'pulsing-danger-border border-red-500' 
          : 'border-cyan-500/20 shadow-cyan-glow'
      }`}
    >
      {/* Scanline pattern overlay */}
      <div className="absolute inset-0 scanline pointer-events-none z-10" />

      {offline ? (
        <div className="flex flex-col items-center gap-4 text-gray-500 select-none p-8 text-center">
          <div className="p-4 rounded-full bg-red-950/20 border border-red-500/20 animate-pulse text-red-500/80">
            <VideoOff size={44} />
          </div>
          <div className="font-orbitron font-semibold tracking-wider text-red-500/80">
            CAMERA FEED OFFLINE
          </div>
          <div className="text-xs max-w-sm text-gray-400 font-mono">
            Ensure backend is active on port 5000 and camera stream is connected.
          </div>
        </div>
      ) : (
        <img
          src={feedUrl}
          alt="Live Camera Feed"
          onError={handleImageError}
          className="w-full h-full object-cover"
        />
      )}

      {/* Badges Overlays */}
      <div className="absolute top-4 left-4 z-20 flex gap-2">
        <div className="flex items-center gap-1.5 px-3 py-1 bg-black/60 backdrop-blur-md rounded-md border border-gray-800 text-[10px] font-mono tracking-wider text-gray-300">
          <span className="h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
          LIVE STREAM
        </div>
        
        {projectorActive && connected && (
          <div className="flex items-center gap-1.5 px-3 py-1 bg-red-950/80 backdrop-blur-md rounded-md border border-red-500/50 text-[10px] font-mono font-bold tracking-wider text-red-400 animate-pulse">
            PROJECTOR ACTIVE
          </div>
        )}
      </div>

      <div className="absolute bottom-4 right-4 z-20 px-2.5 py-1 bg-black/60 backdrop-blur-md rounded-md border border-gray-800 text-[9px] font-mono text-gray-400">
        1280 x 720 @ 30FPS
      </div>
    </div>
  );
}
