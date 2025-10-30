# YOUR CODE HERE
import sys
from collections import deque

def main():
    H, W, K = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize the count of valid paths
    total = 0
    
    # Iterate over all possible starting positions
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                # Initialize the queue with the starting position and the path taken so far
                queue = deque()
                queue.append((i, j, {(i, j)}))
                
                # Perform BFS for K steps
                for step in range(K):
                    next_queue = deque()
                    while queue:
                        current_i, current_j, visited = queue.popleft()
                        for di, dj in directions:
                            ni, nj = current_i + di, current_j + dj
                            if 0 <= ni < H and 0 <= nj < W:
                                if grid[ni][nj] == '.' and (ni, nj) not in visited:
                                    new_visited = set(visited)
                                    new_visited.add((ni, nj))
                                    next_queue.append((ni, nj, new_visited))
                    queue = next_queue
                
                # After K steps, count the number of paths in the queue
                total += len(queue)
    
    print(total)

if __name__ == "__main__":
    main()