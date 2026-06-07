import React from 'react';

export default function ConnectionStatus({ connected }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border border-gray-800 bg-gray-900/60 text-xs font-mono select-none">
      <span className={`relative flex h-2 w-2`}>
        {/* Pulsing effect when disconnected */}
        {!connected && (
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
        )}
        <span
          className={`relative inline-flex rounded-full h-2 w-2 ${
            connected ? 'bg-green-500 shadow-[0_0_8px_#22c55e]' : 'bg-red-500 shadow-[0_0_8px_#ef4444]'
          }`}
        ></span>
      </span>
      <span className={connected ? 'text-green-400' : 'text-red-400 font-bold'}>
        {connected ? 'BACKEND LINK ACTIVE' : 'DISCONNECTED / RETRYING'}
      </span>
    </div>
  );
}
