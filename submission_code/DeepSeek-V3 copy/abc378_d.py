# YOUR CODE HERE
import sys
from collections import deque

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Initialize the count of valid paths
    total = 0
    
    # Directions for moving up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Iterate over all possible starting positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                # Initialize a queue for BFS
                # Each element in the queue is a tuple: (current position, visited set, current path length)
                queue = deque()
                queue.append(((i, j), {(i, j)}, 0))
                
                while queue:
                    (x, y), visited, steps = queue.popleft()
                    if steps == K:
                        total += 1
                        continue
                    for dx, dy in directions:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '.' and (nx, ny) not in visited:
                                new_visited = set(visited)
                                new_visited.add((nx, ny))
                                queue.append(((nx, ny), new_visited, steps + 1))
    
    print(total)

if __name__ == "__main__":
    main()