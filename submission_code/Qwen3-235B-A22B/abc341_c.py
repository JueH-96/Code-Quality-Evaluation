def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    H = int(input[ptr])
    ptr += 1
    W = int(input[ptr])
    ptr += 1
    N = int(input[ptr])
    ptr += 1
    T = input[ptr]
    ptr += 1
    grid = []
    for _ in range(H):
        grid.append(input[ptr])
        ptr += 1
    
    reversed_steps = []
    for c in reversed(T):
        if c == 'L':
            reversed_steps.append('R')
        elif c == 'R':
            reversed_steps.append('L')
        elif c == 'U':
            reversed_steps.append('D')
        elif c == 'D':
            reversed_steps.append('U')
    
    delta = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    
    # Initialize prev_dp
    prev_dp = [[False]*W for _ in range(H)]
    for i in range(H):
        row = grid[i]
        for j in range(W):
            if row[j] == '.':
                prev_dp[i][j] = True
    
    for step in reversed_steps:
        dx, dy = delta[step]
        curr_dp = [[False]*W for _ in range(H)]
        for i in range(H):
            grid_row = grid[i]
            for j in range(W):
                if grid_row[j] != '.':
                    continue
                prev_i = i - dx
                prev_j = j - dy
                if 0 <= prev_i < H and 0 <= prev_j < W:
                    if prev_dp[prev_i][prev_j]:
                        curr_dp[i][j] = True
        prev_dp = curr_dp
    
    count = 0
    for row in prev_dp:
        for cell in row:
            if cell:
                count += 1
    print(count)

if __name__ == '__main__':
    main()