import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    from collections import deque
    
    MOD = 998244353
    
    data = sys.stdin.read().split()
    H, W = map(int, data[:2])
    grid = data[2:]
    
    # comp_id[i][j] = component index for green cell, or -1
    comp_id = [[-1]*W for _ in range(H)]
    comp_count = 0
    # Directions
    dirs = [(1,0),(-1,0),(0,1),(0,-1)]
    
    # BFS to label green components
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#' and comp_id[i][j] == -1:
                # new component
                dq = deque()
                dq.append((i,j))
                comp_id[i][j] = comp_count
                while dq:
                    x,y = dq.popleft()
                    for dx,dy in dirs:
                        nx, ny = x+dx, y+dy
                        if 0 <= nx < H and 0 <= ny < W:
                            if grid[nx][ny] == '#' and comp_id[nx][ny] == -1:
                                comp_id[nx][ny] = comp_count
                                dq.append((nx,ny))
                comp_count += 1
    
    # Count red cells and sum of adjustments
    R = 0
    sum_adj = 0
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                R += 1
                # gather distinct neighboring green component ids
                neigh = set()
                for dx,dy in dirs:
                    nx, ny = i+dx, j+dy
                    if 0 <= nx < H and 0 <= ny < W and grid[nx][ny] == '#':
                        neigh.add(comp_id[nx][ny])
                d = len(neigh)
                if d == 0:
                    sum_adj += 1
                else:
                    sum_adj -= (d - 1)
    # Initial components C = comp_count
    C = comp_count
    # Expected numerator P = R*C + sum_adj, denominator Q = R
    P = (C * R + sum_adj) % MOD
    Q = R % MOD
    # Compute Q inverse
    Q_inv = pow(Q, MOD-2, MOD)
    ans = (P * Q_inv) % MOD
    print(ans)

if __name__ == "__main__":
    main()