import sys
from collections import deque

def main():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Precompute safe matrix
    safe = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if S[i][j] == '.':
                valid = True
                # Check up
                if i > 0 and S[i-1][j] == '#':
                    valid = False
                # Check down
                if i < H-1 and S[i+1][j] == '#':
                    valid = False
                # Check left
                if j > 0 and S[i][j-1] == '#':
                    valid = False
                # Check right
                if j < W-1 and S[i][j+1] == '#':
                    valid = False
                safe[i][j] = valid
            else:
                safe[i][j] = False
    
    visited = [[False]*W for _ in range(H)]
    max_degree = 0
    
    for i in range(H):
        for j in range(W):
            if S[i][j] == '.' and not visited[i][j]:
                queue = deque()
                queue.append((i, j))
                visited[i][j] = True
                count = 1
                while queue:
                    x, y = queue.popleft()
                    if safe[x][y]:
                        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                            nx = x + dx
                            ny = y + dy
                            if 0 <= nx < H and 0 <= ny < W:
                                if not visited[nx][ny] and S[nx][ny] == '.':
                                    visited[nx][ny] = True
                                    count += 1
                                    queue.append((nx, ny))
                if count > max_degree:
                    max_degree = count
    print(max_degree)

if __name__ == "__main__":
    main()