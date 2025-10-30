def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = data[1:N+1]
    
    # Count 'o's in each row and column
    row_counts = [0] * N
    col_counts = [0] * N
    
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                row_counts[i] += 1
                col_counts[j] += 1
    
    total = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                total += (row_counts[i] - 1) * (col_counts[j] - 1)
    
    print(total)

if __name__ == "__main__":
    main()