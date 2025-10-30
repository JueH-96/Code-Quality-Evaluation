from collections import deque
import sys

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    pattern = "snuke"
    
    # Check the starting cell
    if grid[0][0] != 's':
        print("No")
        return
    
    # BFS setup
    dist = [[-1]*W for _ in range(H)]
    dist[0][0] = 0
    dq = deque()
    dq.append((0, 0))
    
    # Directions: up, down, left, right
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    
    while dq:
        i, j = dq.popleft()
        step = dist[i][j]
        next_char = pattern[(step+1) % 5]
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if dist[ni][nj] == -1 and grid[ni][nj] == next_char:
                    dist[ni][nj] = step + 1
                    dq.append((ni, nj))
    
    # Check if we reached bottom-right
    if dist[H-1][W-1] != -1:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()