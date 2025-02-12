import pygame
import cv2
import numpy as np
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.shaders import (
    compileShader,
    glCreateProgram,
    glAttachShader,
    glBindAttribLocation,
    glLinkProgram,
    glGetProgramiv,
    glGetProgramInfoLog,
    glUseProgram,
    GL_LINK_STATUS
)

# Initialize video capture
cap = cv2.VideoCapture("LookingUpAtTheSun.mp4")
width, height = int(cap.get(3)), int(cap.get(4))

# Define adjustable corner points
quad_vertices = np.array([
    [-1,  1,  0],  # Top-left
    [ 1,  1,  0],  # Top-right
    [ 1, -1,  0],  # Bottom-right
    [-1, -1,  0],  # Bottom-left
], dtype=np.float32)

tex_coords = np.array([
    [0, 1], [1, 1], [1, 0], [0, 0]
], dtype=np.float32)

indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

# GLSL 1.20 for older macOS/OpenGL 2.1
VERTEX_SHADER = """
#version 120

attribute vec3 position;
attribute vec2 texCoord;
varying vec2 fragTexCoord;

void main() {
    fragTexCoord = texCoord;
    gl_Position = vec4(position, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 120

varying vec2 fragTexCoord;
uniform sampler2D videoTexture;

void main() {
    gl_FragColor = texture2D(videoTexture, fragTexCoord);
}
"""

pygame.init()

# Create an OpenGL-compatible window
screen = pygame.display.set_mode((width, height), DOUBLEBUF | OPENGL)
glEnable(GL_TEXTURE_2D)

# Compile Shaders
vert_shader = compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
frag_shader = compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)

shader = glCreateProgram()
glAttachShader(shader, vert_shader)
glAttachShader(shader, frag_shader)

# Bind attribute locations before linking
# position -> location 0
# texCoord -> location 1
glBindAttribLocation(shader, 0, "position")
glBindAttribLocation(shader, 1, "texCoord")

# Link the program
glLinkProgram(shader)
link_status = glGetProgramiv(shader, GL_LINK_STATUS)
if not link_status:
    log = glGetProgramInfoLog(shader)
    print("Shader Link Error:\n", log.decode())

# Create Buffers (NO VAO, to support OpenGL 2.1)
VBOs = glGenBuffers(2)
EBO = glGenBuffers(1)

# Fill Position Buffer
glBindBuffer(GL_ARRAY_BUFFER, VBOs[0])
glBufferData(GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, GL_DYNAMIC_DRAW)

# Fill Texture Coordinate Buffer
glBindBuffer(GL_ARRAY_BUFFER, VBOs[1])
glBufferData(GL_ARRAY_BUFFER, tex_coords.nbytes, tex_coords, GL_STATIC_DRAW)

# Fill Index Buffer
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

selected_corner = 0

def update_vertices():
    # Update position buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0])
    glBufferSubData(GL_ARRAY_BUFFER, 0, quad_vertices.nbytes, quad_vertices)

def move_corner(dx, dy):
    global quad_vertices, selected_corner
    quad_vertices[selected_corner][0] += dx
    quad_vertices[selected_corner][1] += dy
    update_vertices()

running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_1:
                selected_corner = 0
            elif event.key == K_2:
                selected_corner = 1
            elif event.key == K_3:
                selected_corner = 2
            elif event.key == K_4:
                selected_corner = 3
            elif event.key == K_LEFT:
                move_corner(-0.05, 0)
            elif event.key == K_RIGHT:
                move_corner(0.05, 0)
            elif event.key == K_UP:
                move_corner(0, 0.05)
            elif event.key == K_DOWN:
                move_corner(0, -0.05)

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    # Flip & convert
    frame = cv2.flip(frame, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Update texture
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame)

    # Clear the screen
    glClear(GL_COLOR_BUFFER_BIT)

    # Use our shader
    glUseProgram(shader)

    # Bind the index buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)

    # Setup position attribute
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0])
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)

    # Setup texture coordinate attribute
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[1])
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)

    # Draw
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
cap.release()

