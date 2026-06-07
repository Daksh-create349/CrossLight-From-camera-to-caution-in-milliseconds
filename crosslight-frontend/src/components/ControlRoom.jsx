import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Eye, ShieldAlert, Cpu } from 'lucide-react';
import ConnectionStatus from './ConnectionStatus';
import CameraFeed from './CameraFeed';
import TrafficLightPanel from './TrafficLightPanel';
import ActiveTracksPanel from './ActiveTracksPanel';
import RiskAlertsPanel from './RiskAlertsPanel';
import GlobalAlert from './GlobalAlert';

export default function ControlRoom({ wsState }) {
  const { tracks, lightState, projectorActive, riskEvents, overrideLog, connected } = wsState;
  const [clock, setClock] = useState(new Date().toLocaleTimeString());

  // Tick clock every second
  useEffect(() => {
    const interval = setInterval(() => {
      setClock(new Date().toLocaleTimeString());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const hasRisk = riskEvents.length > 0;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.98 }}
      transition={{ duration: 0.4 }}
      className="min-h-screen bg-[#050B14] bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,rgba(12,20,40,0.8),rgba(5,11,20,1))] text-gray-100 flex flex-col"
    >
      {/* Global override alert banner */}
      <AnimatePresence>
        {hasRisk && connected && <GlobalAlert active={true} />}
      </AnimatePresence>

      {/* Header bar */}
      <header className="border-b border-gray-800 bg-[#0C1428]/80 backdrop-blur-md px-6 py-4 flex flex-col md:flex-row gap-4 items-center justify-between z-10">
        <div className="flex items-center gap-3 select-none">
          <div className="p-2 rounded-lg bg-cyan-950/40 border border-cyan-500/30 text-accent">
            <Cpu size={20} className="animate-spin-slow" />
          </div>
          <div>
            <h1 className="font-orbitron font-extrabold tracking-widest text-sm md:text-base m-0 text-white flex items-center gap-2">
              CROSSLIGHT <span className="text-accent text-xs font-mono font-light px-1.5 py-0.5 rounded border border-cyan-500/20 bg-cyan-950/20">V2</span>
            </h1>
            <p className="text-[10px] font-mono text-gray-400">Autonomous Collision Avoidance Console</p>
          </div>
        </div>

        {/* Live system clock */}
        <div className="font-mono text-xs md:text-sm tracking-widest bg-gray-900/60 border border-gray-800 px-4 py-1.5 rounded-full text-cyan-400">
          SYS_TIME: {clock}
        </div>

        {/* Connection status component */}
        <ConnectionStatus connected={connected} />
      </header>

      {/* Main interface layout */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-5 gap-6">
        
        {/* Left Column: Live Stream (60% width on desktop) */}
        <div className="lg:col-span-3 flex flex-col gap-4">
          <div className="flex items-center justify-between border-b border-gray-800 pb-2">
            <h2 className="font-orbitron text-xs font-bold tracking-widest text-accent flex items-center gap-2 m-0">
              <Eye size={16} /> LIVE TELEMETRY STREAM
            </h2>
            {hasRisk && connected && (
              <span className="font-mono text-[10px] font-bold text-red-500 border border-red-500/30 bg-red-950/20 px-2 py-0.5 rounded animate-pulse flex items-center gap-1">
                <ShieldAlert size={12} /> SIGNAL OVERRIDE
              </span>
            )}
          </div>
          
          <CameraFeed projectorActive={projectorActive} connected={connected} />
        </div>

        {/* Right Column: Dashboard Info Cards (40% width on desktop) */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <TrafficLightPanel lightState={lightState} projectorActive={projectorActive} />
          
          <ActiveTracksPanel tracks={tracks} />
          
          <RiskAlertsPanel riskEvents={riskEvents} overrideLog={overrideLog} />
        </div>
        
      </main>

      {/* Subtle bottom details */}
      <footer className="border-t border-gray-900/80 bg-black/40 py-3 text-center text-[10px] font-mono text-gray-500 tracking-wider">
        CROSSLIGHT MONITORING SYSTEMS © 2026 // ALL RIGHTS RESERVED
      </footer>
    </motion.div>
  );
}
