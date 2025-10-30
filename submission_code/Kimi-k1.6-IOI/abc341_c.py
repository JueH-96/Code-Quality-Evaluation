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
    
    # Initialize next_dp as dp[N]: all land cells are True
    next_dp = [[False]*W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                next_dp[i][j] = True
    
    # Process each move in reverse order
    for k in range(N-1, -1, -1):
        current_dp = [[False]*W for _ in range(H)]
        move = T[k]
        for i in range(H):
            for j in range(W):
                if grid[i][j] != '.':
                    continue
                # Compute new position after applying move (reverse direction)
                if move == 'L':
                    ni, nj = i, j - 1
                elif move == 'R':
                    ni, nj = i, j + 1
                elif move == 'U':
                    ni, nj = i - 1, j
                else:  # D
                    ni, nj = i + 1, j
                # Check boundaries and next_dp
                if 0 <= ni < H and 0 <= nj < W and next_dp[ni][nj]:
                    current_dp[i][j] = True
        next_dp = current_dp
    
    # Count the number of True in next_dp (dp[0])
    count = 0
    for row in next_dp:
        count += sum(row)
    print(count)

if __name__ == '__main__':
    main()