# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:29:56 2024

@author: Ezequiel
"""

import pygame
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# Configuración del juego
GRID_SIZE = 16
CELL_SIZE = 30
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.6
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.01

# Inicializa Pygame
pygame.init()
window = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + 50))
pygame.display.set_caption("Mini Snake con Aprendizaje por Refuerzo")
clock = pygame.time.Clock()

# Fuentes y colores
FONT = pygame.font.Font(None, 36)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Inicializar tabla Q
def load_q_table(filepath):
    if os.path.exists(filepath):
        Q_table = np.load(filepath)
        print("Tabla Q cargada exitosamente.")
    else:
        Q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        print("No se encontró una tabla Q existente. Se creó una nueva.")
    return Q_table

def save_q_table(Q_table, filepath):
    np.save(filepath, Q_table)
    print("Tabla Q guardada exitosamente.")

# # # Inicializa el entorno
# def reset_game():
#        snake = [(8, 8)]
#        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
#        while food == snake[0]:
#            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
#        return snake, food



# # Inicializa el entorno
def reset_game():
#        # Generar coordenadas aleatorias para la cabeza de la serpiente
        snake_x = random.randint(0, GRID_SIZE - 1)
        snake_y = random.randint(0, GRID_SIZE - 1)
        snake = [(snake_x, snake_y)]

# # #      # Generar coordenadas aleatorias para el alimento
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    
# # #      # Asegurar que el alimento no aparezca en la misma posición que la cabeza de la serpiente
        while food == snake[0]:
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    
        return snake, food


# Mover la serpiente
def move(snake, direction):
    head_x, head_y = snake[0]
    if direction == 'UP':
        head_x -= 1
    elif direction == 'DOWN':
        head_x += 1
    elif direction == 'LEFT':
        head_y -= 1
    elif direction == 'RIGHT':
        head_y += 1
    new_head = (head_x, head_y)
    snake = [new_head] + snake
    return snake
    
    #print(f"Antes de comer/mover: {len(snake)} segmentos")

# Verificar colisiones
def check_collision(snake):
    head = snake[0]
    return (head[0] < 0 or head[0] >= GRID_SIZE or
            head[1] < 0 or head[1] >= GRID_SIZE or
            head in snake[1:])

# Generar nuevo alimento
def place_food(snake):
    food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    while food in snake:
        food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
    return food

# Dibujar el entorno
def draw_window(snake, food, score):
    window.fill(BLACK)
    food_rect = pygame.Rect(food[1] * CELL_SIZE, food[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(window, RED, food_rect)
    for segment in snake:
        segment_rect = pygame.Rect(segment[1] * CELL_SIZE, segment[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(window, GREEN, segment_rect)
    score_text = FONT.render(f"Puntaje: {score}", True, WHITE)
    window.blit(score_text, (10, WINDOW_SIZE + 10))
    pygame.display.update()

# Ruta para guardar la tabla Q
q_table_filepath = "tabla_Q.npy"
Q_table = load_q_table(q_table_filepath)


snake, food = reset_game()
score = 0
consecutive_food = 0  # Contador de comidas consecutivas
running = True
reward_history = []

num_episodes = 2000  # Número de episodios para el entrenamiento inicial

total_comidas = 0


max_length = 0      # Inicializa el contador de longitud máxima
# Bucle de entrenamiento
for episode in range(num_episodes):
    state = (snake[0][0], snake[0][1])
    total_reward = 0
    steps = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if random.uniform(0, 1) < EPSILON:
            action = random.choice(range(len(ACTIONS)))
        else:
            action = np.argmax(Q_table[state[0], state[1]])

        direction = ACTIONS[action]
        new_snake = move(snake, direction)
        reward = -1

        if check_collision(new_snake):
            if new_snake[0] in new_snake[1:]:
                reward = -30  # Penalización mayor por colisión consigo misma
            else:
                reward = -20 # Penalización por colisión con las paredes
            
            consecutive_food = 0  # Reiniciar contador si choca
            if 0 <= new_snake[0][0] < GRID_SIZE and 0 <= new_snake[0][1] < GRID_SIZE:
                new_state = (new_snake[0][0], new_snake[0][1])
                Q_table[state[0], state[1], action] += ALPHA * (
                    reward + GAMMA * np.max(Q_table[new_state[0], new_state[1]]) -
                    Q_table[state[0], state[1], action]
                )
            running = False
        else:
            if new_snake[0] == food:
                reward = 10
                consecutive_food += 1
                if consecutive_food > 1:
                    reward += 30  # Recompensa adicional por comida consecutiva sin morir
                score += 1
                food = place_food(new_snake)
                new_snake.append(snake[-1])
                total_comidas += 1
                max_length = max(max_length, len(new_snake))
            else:
                new_snake.pop()

            if 0 <= new_snake[0][0] < GRID_SIZE and 0 <= new_snake[0][1] < GRID_SIZE:
                new_state = (new_snake[0][0], new_snake[0][1])
                Q_table[state[0], state[1], action] += ALPHA * (
                    reward + GAMMA * np.max(Q_table[new_state[0], new_state[1]]) -
                    Q_table[state[0], state[1], action]
                )

                state = new_state
                snake = new_snake
                total_reward += reward
                draw_window(snake, food, score)
                clock.tick(8)

        steps += 1  # Incrementar el contador de pasos

    reward_history.append(total_reward)
    EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Max Length: {max_length}")
        # Graficar las recompensas y la duración de los episodios
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(reward_history)
        plt.xlabel("Episodios")
        plt.ylabel("Recompensa acumulada")
        plt.title("Recompensa por episodio")
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(reward_history)+1), reward_history)
        plt.xlabel("Episodios")
        plt.ylabel("Duración del episodio")
        plt.title("Duración del episodio por episodio")
        
        plt.show()
        
    snake, food = reset_game()
    score = 0
    running = True

# Guardar la tabla Q al finalizar los episodios
save_q_table(Q_table, q_table_filepath)

pygame.quit()
print(f"Total de comidas al finalizar el entrenamiento: {total_comidas}")
print(f"Longitud máxima alcanzada: {max_length}")
