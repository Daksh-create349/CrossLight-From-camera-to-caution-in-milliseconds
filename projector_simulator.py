import pygame
import numpy as np
import time

class ProjectorSimulator:
    """
    Simulates a projector output using Pygame in a resizable window.
    Draws a visual barrier ("STOP" sign) on the road screen.
    """
    def __init__(self, width=1024, height=768):
        import threading
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.barriers = []
        self.running = False
        
        # Create a resizable screen
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Projector Output")
        self.font = pygame.font.SysFont("Arial", 40, bold=True)
        self.clear()

    def clear(self):
        """Fills the window with black."""
        self.screen.fill((0, 0, 0))

    def project_barrier(self, position, size=(200, 100)):
        """
        Draws a red filled rectangle at 'position' (centered) with white text "STOP" inside.
        Includes a subtle animated pulsing border.
        
        Args:
            position (tuple): (x, y) center position of the barrier.
            size (tuple): (width, height) of the barrier.
        """
        cx, cy = position
        w, h = size
        
        # Center coordinates
        x = int(cx - w / 2)
        y = int(cy - h / 2)
        rect = pygame.Rect(x, y, w, h)
        
        # Draw main red filled rectangle
        pygame.draw.rect(self.screen, (200, 0, 0), rect)
        
        # Subtle animated pulsing border using sine wave over time ticks
        ticks = pygame.time.get_ticks()
        pulse = int(abs(np.sin(ticks * 0.005)) * 8) + 2
        # Pulsing color from orange to red
        g_val = int(100 + abs(np.sin(ticks * 0.005)) * 155)
        border_color = (255, g_val, 0)
        pygame.draw.rect(self.screen, border_color, rect, width=pulse)
        
        # Render white text "STOP" in the center
        text_surf = self.font.render("STOP", True, (255, 255, 255))
        text_rect = text_surf.get_rect(center=(cx, cy))
        self.screen.blit(text_surf, text_rect)

    def update(self):
        """
        Runs a non-blocking Pygame update. Pumps window resize/close events
        and updates the display.
        """
        # Drain events to prevent window from freezing or showing "Not Responding"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pass
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                
        pygame.event.pump()
        pygame.display.flip()

    def set_barriers(self, positions):
        """Thread-safe setter for active barrier positions."""
        with self.lock:
            self.barriers = list(positions)

    def run(self):
        """Indefinite loop for background thread execution."""
        self.running = True
        while self.running:
            self.clear()
            with self.lock:
                current_barriers = list(self.barriers)
            for pos in current_barriers:
                self.project_barrier(pos)
            self.update()
            time.sleep(0.016)  # yield CPU (~60 FPS)

if __name__ == '__main__':
    print("Starting ProjectorSimulator standalone test...")
    proj = ProjectorSimulator()
    
    start_time = time.time()
    print("Projecting barrier at center of screen for 2 seconds...")
    
    while time.time() - start_time < 2.0:
        proj.clear()
        # Draw the barrier at the center of the window
        proj.project_barrier((proj.width // 2, proj.height // 2))
        proj.update()
        time.sleep(0.01) # Yield CPU control
        
    print("Clearing screen for 1 second...")
    proj.clear()
    proj.update()
    time.sleep(1.0)
    
    print("Exiting standalone test.")
    pygame.quit()
