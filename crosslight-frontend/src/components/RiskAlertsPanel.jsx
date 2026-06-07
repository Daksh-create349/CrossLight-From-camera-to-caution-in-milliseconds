import React from 'react';
import { motion } from 'framer-motion';
import { Terminal, Activity } from 'lucide-react';

export default function RiskAlertsPanel({ riskEvents, overrideLog }) {
  // Take last 5 current risk events
  const currentEvents = (riskEvents || []).slice(-5);

  return (
    <div className="glass rounded-2xl p-5 flex flex-col h-[350px] shadow-cyan-glow/5">
      <div className="flex justify-between items-center mb-3 border-b border-gray-800 pb-2">
        <h3 className="font-orbitron font-bold text-xs tracking-widest text-accent">
          RISK ANALYSIS & SIGNAL OVERRIDES
        </h3>
        <span className="font-mono text-[10px] text-red-400 bg-red-950/20 px-2 py-0.5 rounded border border-red-500/20 flex items-center gap-1.5">
          <Activity size={10} className="animate-pulse" />
          ANALYZER ACTIVE
        </span>
      </div>

      <div className="flex-1 flex flex-col gap-4 overflow-hidden">
        {/* Top: Current Risk Events */}
        <div className="flex-1 flex flex-col min-h-[120px]">
          <h4 className="text-[10px] font-mono text-gray-400 uppercase tracking-wider mb-2">
            Active Dangers ({currentEvents.length})
          </h4>
          <div className="flex-1 overflow-y-auto pr-1 flex flex-col gap-2">
            {currentEvents.length === 0 ? (
              <div className="flex-1 flex items-center justify-center text-gray-600 font-mono text-xs select-none">
                No active collision threats detected
              </div>
            ) : (
              currentEvents.map((event, idx) => {
                const conf = event.confidence !== undefined ? event.confidence : 1.0;
                return (
                  <div key={idx} className="bg-red-950/15 border border-red-500/20 rounded-lg p-2.5 flex flex-col gap-1.5">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-red-400 font-bold uppercase">
                        {event.type.replace('_', ' ')}
                      </span>
                      <span className="text-gray-400">
                        Vehicle #{event.vehicle_id || '?'}
                      </span>
                    </div>
                    {/* Confidence bar */}
                    <div className="w-full flex items-center gap-2">
                      <div className="flex-1 h-1.5 bg-gray-900 rounded-full overflow-hidden border border-gray-800">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${conf * 100}%` }}
                          className="h-full bg-red-500 shadow-[0_0_8px_#ef4444]"
                        />
                      </div>
                      <span className="text-[10px] font-mono text-gray-400 w-8 text-right">
                        {(conf * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Bottom: Override Log */}
        <div className="h-[120px] flex flex-col border-t border-gray-800 pt-3">
          <div className="flex justify-between items-center mb-2">
            <h4 className="text-[10px] font-mono text-gray-400 uppercase tracking-wider flex items-center gap-1.5">
              <Terminal size={12} className="text-accent" />
              Override Command History
            </h4>
            <span className="text-[9px] font-mono text-gray-600 uppercase">
              Max 20 logs
            </span>
          </div>
          <div className="flex-1 overflow-y-auto pr-1 font-mono text-[10px] text-gray-500 flex flex-col gap-1 bg-black/30 p-2 rounded-lg border border-gray-900">
            {overrideLog.length === 0 ? (
              <div className="h-full flex items-center justify-center text-gray-700 select-none">
                System history clean
              </div>
            ) : (
              overrideLog.map((log, index) => (
                <div key={index} className="flex gap-2 hover:text-gray-300">
                  <span className="text-accent select-none">&gt;</span>
                  <span className="text-red-400/80 font-semibold">{log}</span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
