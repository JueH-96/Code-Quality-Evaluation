mod = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    H = int(next(it))
    W = int(next(it))
    grid = []
    for _ in range(H):
        row = list(map(int, [next(it) for _ in range(W)]))
        grid.append(row)
    
    f = [[0]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if i == 0 and j == 0:
                f[i][j] = grid[i][j]
            else:
                total = 0
                if i > 0:
                    total = (total + f[i-1][j]) % mod
                if j > 0:
                    total = (total + f[i][j-1]) % mod
                f[i][j] = total * grid[i][j] % mod
    
    Q = int(next(it))
    sh = int(next(it)) - 1
    sw = int(next(it)) - 1
    curr_i, curr_j = sh, sw
    
    out_lines = []
    for _ in range(Q):
        d = next(it)
        a_val = int(next(it))
        if d == 'U':
            curr_i -= 1
        elif d == 'D':
            curr_i += 1
        elif d == 'L':
            curr_j -= 1
        elif d == 'R':
            curr_j += 1
        
        grid[curr_i][curr_j] = a_val
        
        for i in range(curr_i, H):
            start_j = curr_j if i == curr_i else 0
            for j in range(start_j, W):
                if i == 0 and j == 0:
                    total_ways = 1
                else:
                    total_ways = 0
                    if i > 0:
                        total_ways = (total_ways + f[i-1][j]) % mod
                    if j > 0:
                        total_ways = (total_ways + f[i][j-1]) % mod
                f[i][j] = total_ways * grid[i][j] % mod
        
        out_lines.append(str(f[H-1][W-1]))
    
    print("
".join(out_lines))

if __name__ == "__main__":
    main()