# YOUR CODE HERE
import sys
from collections import deque

def main():
    snuke = ['s','n','u','k','e']
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    if grid[0][0] != snuke[0]:
        print("No")
        return
    
    visited = [[ [False]*5 for _ in range(W)] for _ in range(H)]
    queue = deque()
    queue.append( (0, 0, 0) )
    visited[0][0][0] = True
    
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    
    while queue:
        i, j, step = queue.popleft()
        if i == H-1 and j == W-1:
            print("Yes")
            return
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                next_step = (step +1) %5
                if not visited[ni][nj][next_step] and grid[ni][nj] == snuke[next_step]:
                    visited[ni][nj][next_step] = True
                    queue.append( (ni, nj, next_step) )
    print("No")

if __name__ == "__main__":
    main()