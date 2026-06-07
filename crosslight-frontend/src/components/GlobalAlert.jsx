import React from 'react';
import { motion } from 'framer-motion';

export default function GlobalAlert({ active }) {
  if (!active) return null;

  return (
    <motion.div
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      exit={{ y: -100, opacity: 0 }}
      transition={{ type: 'spring', stiffness: 120, damping: 14 }}
      className="fixed top-0 left-0 right-0 z-50 p-4 flex justify-center pointer-events-none"
    >
      <div className="glass-danger px-8 py-4 rounded-xl shadow-red-glow border-red-500/40 text-center flex items-center gap-3 animate-pulse pointer-events-auto">
        <span className="text-xl">⚠️</span>
        <span className="font-orbitron font-bold tracking-widest text-red-500 text-sm md:text-base">
          COLLISION RISK DETECTED – SIGNAL OVERRIDE ACTIVE
        </span>
      </div>
    </motion.div>
  );
}
