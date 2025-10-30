import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]
    A, B, C, D = map(int, sys.stdin.readline().split())
    
    # Convert to 0-based indexing
    A -= 1
    B -= 1
    C -= 1
    D -= 1
    
    INF = H * W + 1
    dist = [[INF] * W for _ in range(H)]
    dist[A][B] = 0
    dq = deque()
    dq.append((A, B))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while dq:
        x, y = dq.popleft()
        if x == C and y == D:
            print(dist[x][y])
            return
        
        current_k = dist[x][y]
        
        # Move to adjacent cells (no kick)
        for dx, dy in directions:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < H and 0 <= ny < W:
                if S[nx][ny] == '.' and dist[nx][ny] > current_k:
                    dist[nx][ny] = current_k
                    dq.appendleft((nx, ny))
        
        # Kick in each direction for steps 1 and 2
        for dx, dy in directions:
            for step in [1, 2]:
                nx = x + dx * step
                ny = y + dy * step
                if 0 <= nx < H and 0 <= ny < W:
                    if S[nx][ny] == '#' and dist[nx][ny] > current_k + 1:
                        dist[nx][ny] = current_k + 1
                        dq.append((nx, ny))
    
    # According to problem statement, there is always a solution
    print(-1)

if __name__ == "__main__":
    main()