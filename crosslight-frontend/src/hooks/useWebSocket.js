import { useState, useEffect, useRef } from 'react';

export default function useWebSocket() {
  const [tracks, setTracks] = useState([]);
  const [lightState, setLightState] = useState('unknown');
  const [projectorActive, setProjectorActive] = useState(false);
  const [riskEvents, setRiskEvents] = useState([]);
  const [overrideLog, setOverrideLog] = useState([]);
  const [connected, setConnected] = useState(false);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  useEffect(() => {
    function connect() {
      // Connect to backend websocket
      const wsUrl = 'ws://localhost:5000/ws';
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('[WS] Connected to CrossLight Backend.');
        setConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          
          const newTracks = data.tracks || [];
          const newLightState = data.light_state || 'unknown';
          const newProjectorActive = !!data.projector_active;
          const newRiskEvents = data.risk_events || [];

          setTracks(newTracks);
          setLightState(newLightState);
          setProjectorActive(newProjectorActive);
          setRiskEvents(newRiskEvents);

          // If there are any active risk events, prepend a timestamp log entry
          if (newRiskEvents.length > 0) {
            setOverrideLog((prev) => {
              const timestamp = new Date().toLocaleTimeString();
              const entry = `[${timestamp}] Override active - ${newRiskEvents.length} alert(s) detected`;
              // Prevent duplicate consecutive messages for same timestamp
              if (prev.length > 0 && prev[0].includes(`[${timestamp}]`)) {
                return prev;
              }
              const updated = [entry, ...prev];
              return updated.slice(0, 20); // Keep max 20 entries
            });
          }
        } catch (err) {
          console.error('[WS] Failed parsing incoming message:', err);
        }
      };

      ws.onclose = () => {
        console.log('[WS] Connection closed. Attempting reconnect in 3s...');
        setConnected(false);
        scheduleReconnect();
      };

      ws.onerror = (err) => {
        console.error('[WS] WebSocket error:', err);
        ws.close();
      };
    }

    function scheduleReconnect() {
      if (reconnectTimeoutRef.current) return;
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectTimeoutRef.current = null;
        connect();
      }, 3000);
    }

    connect();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return {
    tracks,
    lightState,
    projectorActive,
    riskEvents,
    overrideLog,
    connected,
  };
}
