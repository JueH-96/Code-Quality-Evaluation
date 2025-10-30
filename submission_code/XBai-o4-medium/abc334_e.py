import sys
from collections import deque

MOD = 998244353

def main():
    h, w = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(h)]
    
    component = [[-1] * w for _ in range(h)]
    current_id = 0
    
    # BFS to find connected components of green cells
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '#' and component[i][j] == -1:
                q = deque()
                q.append((i, j))
                component[i][j] = current_id
                while q:
                    x, y = q.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            if grid[nx][ny] == '#' and component[nx][ny] == -1:
                                component[nx][ny] = current_id
                                q.append((nx, ny))
                current_id += 1
    
    original_C = current_id
    
    sum_val = 0
    total_red = 0
    
    # Calculate sum_val and total_red
    for i in range(h):
        for j in range(w):
            if grid[i][j] == '.':
                total_red += 1
                s = set()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < h and 0 <= nj < w:
                        if grid[ni][nj] == '#':
                            s.add(component[ni][nj])
                m = len(s)
                sum_val += (original_C + 1 - m)
    
    # Compute the final answer using modular inverse
    ans = (sum_val % MOD) * pow(total_red, MOD - 2, MOD)
    ans %= MOD
    print(ans)

if __name__ == "__main__":
    main()