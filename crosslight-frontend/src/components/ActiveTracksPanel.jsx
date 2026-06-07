import React from 'react';
import { Shield, ShieldAlert, Navigation } from 'lucide-react';

export default function ActiveTracksPanel({ tracks }) {
  const getBadgeColor = (className) => {
    switch ((className || '').toLowerCase()) {
      case 'person':
        return 'bg-green-950/40 text-green-400 border-green-500/30';
      case 'bus':
        return 'bg-orange-950/40 text-orange-400 border-orange-500/30';
      case 'truck':
        return 'bg-purple-950/40 text-purple-400 border-purple-500/30';
      case 'motorcycle':
        return 'bg-cyan-950/40 text-cyan-400 border-cyan-500/30';
      default:
        return 'bg-blue-950/40 text-blue-400 border-blue-500/30';
    }
  };

  const getSpeedRowStyle = (speed) => {
    if (speed > 80) return 'bg-red-950/30 border-red-500/30 text-red-200';
    if (speed > 50) return 'bg-yellow-950/20 border-yellow-500/20 text-yellow-200';
    return 'hover:bg-gray-900/40 border-gray-800/40';
  };

  return (
    <div className="glass rounded-2xl p-5 flex flex-col h-[280px] shadow-cyan-glow/5">
      <div className="flex justify-between items-center mb-3 border-b border-gray-800 pb-2">
        <h3 className="font-orbitron font-bold text-xs tracking-widest text-accent">
          ACTIVE VEHICLE TRACKS
        </h3>
        <span className="font-mono text-[10px] text-gray-500 bg-gray-900/80 px-2 py-0.5 rounded border border-gray-800">
          COUNT: {tracks.length}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto pr-1">
        {tracks.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center gap-2 text-gray-500 select-none">
            <Shield size={28} className="text-gray-600 animate-pulse" />
            <span className="font-mono text-xs">No active tracks detected</span>
          </div>
        ) : (
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="text-[10px] text-gray-500 border-b border-gray-800 pb-1">
                <th className="py-1.5 font-medium">ID</th>
                <th className="py-1.5 font-medium">TYPE</th>
                <th className="py-1.5 font-medium text-right font-semibold">SPEED</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/30">
              {tracks.map((track) => {
                const speed = track.speed_kmh || 0;
                return (
                  <tr key={track.track_id} className={`border-b border-gray-800/30 ${getSpeedRowStyle(speed)}`}>
                    <td className="py-2.5 font-semibold text-gray-300">
                      #{track.track_id}
                    </td>
                    <td className="py-2.5">
                      <span className={`px-2.5 py-0.5 rounded-full border text-[10px] uppercase font-semibold ${getBadgeColor(track.class)}`}>
                        {track.class}
                      </span>
                    </td>
                    <td className="py-2.5 text-right font-bold flex items-center justify-end gap-1.5">
                      {speed > 50 && (
                        <ShieldAlert size={14} className={speed > 80 ? 'text-red-500 animate-pulse' : 'text-yellow-500'} />
                      )}
                      <span className={speed > 80 ? 'text-red-400' : speed > 50 ? 'text-yellow-400' : 'text-cyan-400'}>
                        {speed.toFixed(1)} km/h
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
