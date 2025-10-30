# YOUR CODE HERE
from collections import deque

def bfs(grid, start, goal, H, W, T):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    queue = deque([(start[0], start[1], 0, set())])  # (x, y, moves, candies_visited)
    visited = set()
    visited.add((start[0], start[1], 0))
    max_candies = -1
    
    while queue:
        x, y, moves, candies_visited = queue.popleft()
        
        if (x, y) == goal:
            max_candies = max(max_candies, len(candies_visited))
            continue
        
        if moves >= T:
            continue
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] != '#':
                new_candies_visited = candies_visited.copy()
                if grid[nx][ny] == 'o':
                    new_candies_visited.add((nx, ny))
                if (nx, ny, moves + 1) not in visited:
                    visited.add((nx, ny, moves + 1))
                    queue.append((nx, ny, moves + 1, new_candies_visited))
    
    return max_candies

def main():
    import sys
    input = sys.stdin.read().split()
    H, W, T = map(int, input[:3])
    grid = input[3:]
    
    start = None
    goal = None
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
    
    result = bfs(grid, start, goal, H, W, T)
    print(result)

if __name__ == "__main__":
    main()