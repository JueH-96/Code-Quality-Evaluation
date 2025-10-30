# YOUR CODE HERE
from collections import deque

def can_reach_goal(H, W, grid, medicines):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    start = None
    goal = None
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    
    if start is None or goal is None:
        return "No"
    
    medicine_map = {}
    for r, c, e in medicines:
        medicine_map[(r-1, c-1)] = e
    
    queue = deque([(start[0], start[1], 0)])
    visited = set()
    visited.add((start[0], start[1], 0))
    
    while queue:
        x, y, energy = queue.popleft()
        
        if (x, y) == goal:
            return "Yes"
        
        if (x, y) in medicine_map:
            new_energy = medicine_map[(x, y)]
            if (x, y, new_energy) not in visited:
                visited.add((x, y, new_energy))
                queue.append((x, y, new_energy))
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#' and energy > 0:
                if (nx, ny, energy - 1) not in visited:
                    visited.add((nx, ny, energy - 1))
                    queue.append((nx, ny, energy - 1))
    
    return "No"

import sys
input = sys.stdin.read().split()
H, W = int(input[0]), int(input[1])
grid = input[2:H+2]
N = int(input[H+2])
medicines = [(int(input[H+3 + 3*i]), int(input[H+4 + 3*i]), int(input[H+5 + 3*i])) for i in range(N)]

print(can_reach_goal(H, W, grid, medicines))