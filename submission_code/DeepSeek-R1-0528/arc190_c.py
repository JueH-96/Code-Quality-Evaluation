mod = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    H = int(data[0])
    W = int(data[1])
    idx = 2
    grid = []
    for i in range(H):
        row = list(map(int, data[idx:idx+W]))
        idx += W
        grid.append(row)
    
    Q = int(data[idx]); idx += 1
    sh = int(data[idx]); sw = int(data[idx+1]); idx += 2
    current_i = sh - 1
    current_j = sw - 1
    
    F0 = [[0] * W for _ in range(H)]
    G0 = [[0] * W for _ in range(H)]
    
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                F0[i][j] = 1
            else:
                up = 0
                if i > 0:
                    up = F0[i-1][j] * grid[i-1][j] % mod
                left = 0
                if j > 0:
                    left = F0[i][j-1] * grid[i][j-1] % mod
                F0[i][j] = (up + left) % mod
                
    for i in range(H-1, -1, -1):
        for j in range(W-1, -1, -1):
            if i == H-1 and j == W-1:
                G0[i][j] = 1
            else:
                down = 0
                if i < H-1:
                    down = grid[i+1][j] * G0[i+1][j] % mod
                right = 0
                if j < W-1:
                    right = grid[i][j+1] * G0[i][j+1] % mod
                G0[i][j] = (down + right) % mod
                
    T = F0[H-1][W-1] * grid[H-1][W-1] % mod
    
    out_lines = []
    for _ in range(Q):
        d = data[idx]; idx += 1
        a_val = int(data[idx]); idx += 1
        if d == 'U':
            new_i = current_i - 1
            new_j = current_j
        elif d == 'D':
            new_i = current_i + 1
            new_j = current_j
        elif d == 'L':
            new_i = current_i
            new_j = current_j - 1
        elif d == 'R':
            new_i = current_i
            new_j = current_j + 1
        i0 = new_i
        j0 = new_j
        old_val = grid[i0][j0]
        change = (a_val - old_val) % mod
        change = change * F0[i0][j0] % mod
        change = change * G0[i0][j0] % mod
        T = (T + change) % mod
        grid[i0][j0] = a_val
        
        for i in range(i0, H):
            for j in range(j0, W):
                if i == i0 and j == j0:
                    continue
                up = 0
                if i > 0:
                    up = F0[i-1][j] * grid[i-1][j] % mod
                left = 0
                if j > 0:
                    left = F0[i][j-1] * grid[i][j-1] % mod
                F0[i][j] = (up + left) % mod
                
        for i in range(i0, -1, -1):
            for j in range(j0, -1, -1):
                if i == i0 and j == j0:
                    continue
                down = 0
                if i < H-1:
                    down = grid[i+1][j] * G0[i+1][j] % mod
                right = 0
                if j < W-1:
                    right = grid[i][j+1] * G0[i][j+1] % mod
                G0[i][j] = (down + right) % mod
                
        current_i = i0
        current_j = j0
        out_lines.append(str(T))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()