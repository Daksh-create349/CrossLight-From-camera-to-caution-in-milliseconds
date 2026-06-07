import React from 'react';
import { motion } from 'framer-motion';

export default function TrafficLightPanel({ lightState, projectorActive }) {
  // Normalize lightState to handle uppercase/lowercase
  const state = (lightState || 'unknown').toLowerCase();

  const lights = [
    { key: 'red', colorClass: 'bg-red-600', shadowClass: 'shadow-red-glow border-red-500/60' },
    { key: 'yellow', colorClass: 'bg-yellow-500', shadowClass: 'shadow-yellow-glow border-yellow-400/60' },
    { key: 'green', colorClass: 'bg-green-500', shadowClass: 'shadow-green-glow border-green-400/60' },
  ];

  return (
    <div className="glass rounded-2xl p-6 flex flex-col items-center justify-between min-h-[160px] shadow-cyan-glow/5">
      <div className="w-full flex justify-between items-center mb-4 border-b border-gray-800 pb-2">
        <h3 className="font-orbitron font-bold text-xs tracking-widest text-accent">
          SIGNAL MONITOR
        </h3>
        <span className="font-mono text-[10px] text-gray-500 uppercase">
          HSV Classifier
        </span>
      </div>

      {/* Traffic Lights vertical/horizontal row */}
      <div className="flex gap-6 items-center justify-center py-2">
        {lights.map((light) => {
          const isActive = state === light.key;
          return (
            <div key={light.key} className="flex flex-col items-center gap-1.5">
              <motion.div
                animate={{
                  opacity: isActive ? 1.0 : 0.15,
                  scale: isActive ? 1.1 : 0.95,
                }}
                transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                className={`w-12 h-12 rounded-full border border-transparent transition-all duration-300 ${light.colorClass} ${
                  isActive ? `${light.shadowClass} border-opacity-100` : ''
                }`}
              />
              <span className={`text-[9px] font-mono tracking-wider ${isActive ? 'text-gray-200 font-bold' : 'text-gray-600'}`}>
                {light.key.toUpperCase()}
              </span>
            </div>
          );
        })}
      </div>

      {/* Projector Override Status */}
      <div className="w-full flex justify-between items-center border-t border-gray-800 pt-3 mt-3 text-xs font-mono">
        <span className="text-gray-400">Intersection Override:</span>
        <div className="flex items-center gap-1.5">
          {projectorActive ? (
            <>
              <span className="h-2 w-2 rounded-full bg-red-500 animate-ping" />
              <span className="text-red-500 font-bold uppercase animate-pulse">ACTIVE OVERRIDE</span>
            </>
          ) : (
            <>
              <span className="h-1.5 w-1.5 rounded-full bg-gray-600" />
              <span className="text-gray-500 uppercase">SYSTEM IDLE</span>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
