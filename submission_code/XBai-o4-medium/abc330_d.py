def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    grid = []
    for _ in range(N):
        grid.append(input[idx])
        idx += 1
    
    row_count = [0] * N
    for i in range(N):
        row_count[i] = grid[i].count('o')
    
    col_count = [0] * N
    for j in range(N):
        cnt = 0
        for i in range(N):
            if grid[i][j] == 'o':
                cnt += 1
        col_count[j] = cnt
    
    ans = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                ans += (row_count[i] - 1) * (col_count[j] - 1)
    
    print(ans)

if __name__ == '__main__':
    main()