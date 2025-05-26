import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import sys
import os
from PIL import Image

# Set window size
WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 810

class Light:
    def __init__(self):
        self.ambient = [0.2, 0.2, 0.2, 0.2]
        self.diffuse = [0.7, 0.7, 0.7, 0.7]
        self.specular = [0.4, 0.4, 0.4, 0.4]
        self.position = [0.0, 0.0, 1.0, 1.0]
        self.id = GL_LIGHT0

    def apply(self):
        glLightfv(self.id, GL_AMBIENT, self.ambient)
        glLightfv(self.id, GL_DIFFUSE, self.diffuse)
        glLightfv(self.id, GL_SPECULAR, self.specular)
        glLightfv(self.id, GL_POSITION, self.position)
        glEnable(self.id)

def read_frame(fname):
    particles = []
    try:
        with open(fname, 'r') as f:
            print(f"reading {fname}")
            n_parts = int(f.readline())
            for line in f:
                values = line.strip().split()
                if len(values) >= 6: 
                    particles.append([float(values[0]), float(values[1]), float(values[2])])
        print("inputfile read")
        return particles
    except:
        return []

def init_gl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glEnable(GL_NORMALIZE)
    
    # Light settings
    light = Light()
    light.apply()

def draw_particles(particles):
    glPushMatrix()
    
    # Material settings (blue-cyan)
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0.0, 0.5, 1.0, 1.0])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0.0, 0.5, 1.0, 1.0])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0.0, 0.5, 1.0, 1.0])

    # Draw each particle
    for p in particles:
        glPushMatrix()
        glTranslatef(p[0], p[1], p[2])
        sphere = gluNewQuadric()
        gluSphere(sphere, 0.0025, 10, 10)
        gluDeleteQuadric(sphere)
        glPopMatrix()

    glPopMatrix()

def save_image(frame_num):
    # Create output directory
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get image data directly from OpenGL framebuffer
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    data = glReadPixels(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(data, dtype=np.uint8)
    image = image.reshape(WINDOW_HEIGHT, WINDOW_WIDTH, 3)
    
    # Flip image vertically (due to difference between OpenGL and image coordinates)
    image = np.flipud(image)
    
    # Convert Numpy array to PIL Image
    image = Image.fromarray(image)
    
    # Save image to output directory
    image.save(os.path.join(output_dir, f'frame_{frame_num:03d}.png'))

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print(f"Usage: {sys.argv[0]} <frame_pattern> [file|block]")
        return

    frame_pattern = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) == 3 else "file"
    current_frame = 0
    frames = []

    # Initialize Pygame
    pygame.init()
    pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Particle Viewer")
    
    # Initialize OpenGL
    init_gl()

    # Camera settings
    glMatrixMode(GL_PROJECTION)
    gluPerspective(60, (WINDOW_WIDTH/WINDOW_HEIGHT), 0.01, 20.0)
    
    glMatrixMode(GL_MODELVIEW)

    if mode == "file":
        gluLookAt(0.0, 0.5, 0.0,
                  0.0, 0.0, 0.0,
                  0.0, 0.0, 1.0)
        
    elif mode == "block":
        gluLookAt(1.0, -1.2, 0.7,
                  0.375, 0.05, 0.275,
                  0.0, 0.0, 1.0)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Read new frame
        if current_frame >= len(frames):
            fname = frame_pattern % current_frame
            particles = read_frame(fname)
            if particles:
                frames.append(particles)
            else:
                running = False
                break

        # Clear screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw particles
        draw_particles(frames[current_frame])
        
        # Save image
        save_image(current_frame)
        
        pygame.display.flip()
        current_frame += 1

    pygame.quit()

if __name__ == "__main__":
    main() 