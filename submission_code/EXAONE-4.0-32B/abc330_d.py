def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    grid = data[1:1+n]
    
    row_count = [0] * n
    col_count = [0] * n
    
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'o':
                row_count[i] += 1
                col_count[j] += 1
                
    ans = 0
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'o':
                a = row_count[i] - 1
                b = col_count[j] - 1
                ans += a * b
                
    print(ans)

if __name__ == "__main__":
    main()