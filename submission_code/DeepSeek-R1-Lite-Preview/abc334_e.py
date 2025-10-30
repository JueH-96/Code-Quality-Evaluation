import sys
from collections import deque

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    grid = input[2:2+H]

    # Initialize component IDs
    comp_id = [[-1 for _ in range(W)] for _ in range(H)]
    
    # Directions: up, down, left, right
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]
    
    # Find connected components
    C = 0
    current_id = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and comp_id[i][j] == -1:
                current_id += 1
                # BFS
                queue = deque()
                queue.append((i,j))
                comp_id[i][j] = current_id
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in dirs:
                        nx = x + dx
                        ny = y + dy
                        if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#' and comp_id[nx][ny] == -1:
                            comp_id[nx][ny] = current_id
                            queue.append((nx,ny))
                C += 1
    
    # Now, iterate through red cells
    S = 0
    R = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                R += 1
                # Find adjacent green cells
                unique_ids = set()
                for dx, dy in dirs:
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '#':
                        unique_ids.add(comp_id[ni][nj])
                k = len(unique_ids)
                new_components = C - k + 1
                S += new_components
    
    # Compute expected value as S / R
    # Compute R = S * R^{-1} mod MOD
    if R == 0:
        print(0)
        return
    R_inv = pow(R, MOD - 2, MOD)
    answer = (S * R_inv) % MOD
    print(answer)

if __name__ == '__main__':
    main()