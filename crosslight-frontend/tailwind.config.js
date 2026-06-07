/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#050B14',
        panel: '#0C1428',
        accent: '#00E5FF',
        danger: '#FF1744',
      },
      fontFamily: {
        orbitron: ['Orbitron', 'sans-serif'],
        mono: ['Courier New', 'Courier', 'monospace'],
      },
      boxShadow: {
        'cyan-glow': '0 0 15px rgba(0, 229, 255, 0.4)',
        'red-glow': '0 0 15px rgba(255, 23, 68, 0.5)',
        'yellow-glow': '0 0 15px rgba(234, 179, 8, 0.5)',
        'green-glow': '0 0 15px rgba(34, 197, 94, 0.5)',
      }
    },
  },
  plugins: [],
}
