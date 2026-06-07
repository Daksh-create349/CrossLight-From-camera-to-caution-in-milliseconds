import React, { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Canvas, useFrame } from '@react-three/fiber';
import { Sparkles } from '@react-three/drei';
import { motion } from 'framer-motion';
import { ArrowRight, ShieldCheck } from 'lucide-react';

// Turntable Car Model built with Three.js primitives (highly reliable & offline-safe)
function CyberCar() {
  const groupRef = useRef();

  useFrame((state, delta) => {
    if (groupRef.current) {
      // Slow rotation on yaw axis
      groupRef.current.rotation.y += delta * 0.35;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Lower chassis */}
      <mesh position={[0, 0.35, 0]}>
        <boxGeometry args={[3.2, 0.4, 1.7]} />
        <meshStandardMaterial color="#0a1220" roughness={0.25} metalness={0.8} />
      </mesh>

      {/* Cabin with transparent glowing material */}
      <mesh position={[-0.2, 0.75, 0]}>
        <boxGeometry args={[1.8, 0.5, 1.4]} />
        <meshStandardMaterial color="#00E5FF" roughness={0.1} metalness={0.9} transparent opacity={0.5} />
      </mesh>

      {/* Futuristic neon light bars */}
      <mesh position={[0, 0.56, 0.86]}>
        <boxGeometry args={[2.6, 0.04, 0.02]} />
        <meshBasicMaterial color="#00E5FF" />
      </mesh>
      <mesh position={[0, 0.56, -0.86]}>
        <boxGeometry args={[2.6, 0.04, 0.02]} />
        <meshBasicMaterial color="#00E5FF" />
      </mesh>

      {/* Wheels */}
      {/* FL */}
      <mesh position={[1.0, 0.25, 0.9]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.38, 0.38, 0.25, 16]} />
        <meshStandardMaterial color="#060910" roughness={0.7} />
      </mesh>
      {/* FR */}
      <mesh position={[1.0, 0.25, -0.9]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.38, 0.38, 0.25, 16]} />
        <meshStandardMaterial color="#060910" roughness={0.7} />
      </mesh>
      {/* RL */}
      <mesh position={[-1.0, 0.25, 0.9]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.38, 0.38, 0.25, 16]} />
        <meshStandardMaterial color="#060910" roughness={0.7} />
      </mesh>
      {/* RR */}
      <mesh position={[-1.0, 0.25, -0.9]} rotation={[Math.PI / 2, 0, 0]}>
        <cylinderGeometry args={[0.38, 0.38, 0.25, 16]} />
        <meshStandardMaterial color="#060910" roughness={0.7} />
      </mesh>

      {/* Pulsing AI object detection wireframe box */}
      <DetectionWireframe />
    </group>
  );
}

function DetectionWireframe() {
  const wireRef = useRef();

  useFrame(({ clock }) => {
    if (wireRef.current) {
      const t = clock.getElapsedTime();
      const scale = 1.05 + Math.sin(t * 3) * 0.015;
      wireRef.current.scale.set(scale, scale, scale);
      wireRef.current.material.opacity = 0.2 + Math.abs(Math.sin(t * 3)) * 0.55;
    }
  });

  return (
    <mesh ref={wireRef}>
      <boxGeometry args={[3.5, 1.25, 2.0]} />
      <meshBasicMaterial
        color="#00E5FF"
        wireframe
        transparent
        opacity={0.5}
      />
    </mesh>
  );
}

// Danger Zone Polygon (trapezoid ahead of the vehicle on the road)
function DangerZone() {
  const zoneRef = useRef();

  useFrame(({ clock }) => {
    if (zoneRef.current) {
      const t = clock.getElapsedTime();
      zoneRef.current.material.opacity = 0.12 + Math.abs(Math.sin(t * 2.2)) * 0.22;
    }
  });

  return (
    <group position={[2.8, 0.01, 0]}>
      {/* Solid trapezoid overlay */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[2.4, 2.8]} />
        <meshBasicMaterial
          ref={zoneRef}
          color="#FF1744"
          transparent
          opacity={0.25}
          side={2}
        />
      </mesh>
      {/* Outlined border */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.002, 0]}>
        <planeGeometry args={[2.4, 2.8]} />
        <meshBasicMaterial
          color="#FF1744"
          wireframe
          transparent
          opacity={0.8}
          side={2}
        />
      </mesh>
    </group>
  );
}

// Asphalt Road Plane
function Road() {
  return (
    <group position={[0, -0.01, 0]}>
      {/* Main road plane */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[25, 8]} />
        <meshStandardMaterial color="#080e1a" roughness={0.85} />
      </mesh>
      
      {/* Center Dashed White Line */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <planeGeometry args={[25, 0.08]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.1} />
      </mesh>
    </group>
  );
}

export default function LandingPage() {
  const navigate = useNavigate();

  return (
    <div className="relative w-full h-screen bg-[#050B14] overflow-hidden flex flex-col items-center justify-between">
      
      {/* 3D Background Canvas */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [4.5, 2.5, 6.5], fov: 42 }}>
          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <pointLight position={[5, 5, 5]} intensity={1.2} />
          <directionalLight position={[-4, 8, 2]} intensity={0.8} />
          <spotLight position={[0, 10, 0]} angle={0.4} penumbra={1} intensity={1.5} color="#00E5FF" castShadow />
          
          {/* Cyber turntable scene components */}
          <CyberCar />
          <DangerZone />
          <Road />

          {/* Particle Sparkles rising up */}
          <Sparkles count={45} scale={8} size={2.5} speed={0.8} color="#00E5FF" position={[0, 1.5, 0]} />
        </Canvas>
      </div>

      {/* Decorative Grid Scanline Overlay */}
      <div className="absolute inset-0 scanline pointer-events-none z-10 opacity-60" />

      {/* Top Brand Tag */}
      <div className="z-20 mt-8 flex items-center gap-2 font-mono text-[10px] tracking-widest text-accent/80 select-none bg-black/40 backdrop-blur-md px-4 py-2 rounded-full border border-cyan-500/20">
        <ShieldCheck size={14} className="animate-pulse text-accent" />
        SECURE AUTOMATED INTERSECTION GUARD
      </div>

      {/* Main Cinematic Centered Card */}
      <div className="z-20 flex flex-col items-center text-center max-w-xl px-6 mb-12">
        <motion.h1
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="font-orbitron font-black text-5xl md:text-7xl tracking-[0.25em] text-transparent bg-clip-text bg-gradient-to-r from-white via-cyan-300 to-blue-500 glow-text mb-4"
        >
          CROSSLIGHT
        </motion.h1>

        <motion.p
          initial={{ y: 30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.2, ease: 'easeOut' }}
          className="font-orbitron text-xs md:text-sm tracking-[0.4em] uppercase text-gray-300 mb-8 font-light"
        >
          Intelligent Collision Prevention System
        </motion.p>

        <motion.button
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => navigate('/control')}
          className="group relative px-8 py-3.5 rounded-full border border-cyan-500/40 bg-cyan-950/20 font-orbitron font-bold text-xs tracking-widest text-accent overflow-hidden shadow-cyan-glow hover:shadow-[0_0_25px_rgba(0,229,255,0.7)] transition-all duration-300 cursor-pointer flex items-center gap-2"
        >
          ENTER CONTROL ROOM
          <ArrowRight size={14} className="group-hover:translate-x-1.5 transition-transform duration-300" />
        </motion.button>
      </div>

      {/* Decorative Technical Readout Details */}
      <div className="z-20 mb-8 flex gap-8 font-mono text-[9px] text-gray-500 tracking-wider select-none">
        <span>LATENCY: &lt;5MS</span>
        <span>YOLOv8n: RUNNING</span>
        <span>FPS: 30.0</span>
      </div>

    </div>
  );
}
