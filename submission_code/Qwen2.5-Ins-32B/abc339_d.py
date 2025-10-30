import sys
from collections import deque

def read_input():
    N = int(input())
    grid = [input().strip() for _ in range(N)]
    return N, grid

def find_players(grid):
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'P':
                yield (i, j)

def bfs(N, grid, start1, start2):
    queue = deque([(start1, start2, 0)])
    visited = set([(start1, start2)])
    
    while queue:
        (x1, y1), (x2, y2), moves = queue.popleft()
        
        if (x1, y1) == (x2, y2):
            return moves
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_x1, new_y1 = x1 + dx, y1 + dy
            new_x2, new_y2 = x2 + dx, y2 + dy
            
            if 0 <= new_x1 < N and 0 <= new_y1 < N and grid[new_x1][new_y1] != '#':
                if 0 <= new_x2 < N and 0 <= new_y2 < N and grid[new_x2][new_y2] != '#':
                    if ((new_x1, new_y1), (new_x2, new_y2)) not in visited:
                        visited.add(((new_x1, new_y1), (new_x2, new_y2)))
                        queue.append(((new_x1, new_y1), (new_x2, new_y2), moves + 1))
    
    return -1

def main():
    N, grid = read_input()
    start1, start2 = find_players(grid)
    result = bfs(N, grid, start1, start2)
    print(result)

if __name__ == "__main__":
    main()