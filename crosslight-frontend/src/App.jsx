import React from 'react';
import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import useWebSocket from './hooks/useWebSocket';
import LandingPage from './components/LandingPage';
import ControlRoom from './components/ControlRoom';

function AnimatedRoutes() {
  const location = useLocation();
  const wsState = useWebSocket(); // Fetch live websocket data from backend

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<LandingPage />} />
        <Route path="/control" element={<ControlRoom wsState={wsState} />} />
      </Routes>
    </AnimatePresence>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AnimatedRoutes />
    </BrowserRouter>
  );
}
