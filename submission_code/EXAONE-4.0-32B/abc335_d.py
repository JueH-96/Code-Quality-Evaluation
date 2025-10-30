def main():
    N = int(input().strip())
    grid = [[None] * N for _ in range(N)]
    mid = N // 2
    grid[mid][mid] = 'T'
    num = 1
    for k in range(mid):
        L = N - 2 * k
        top = k
        bottom = k + L - 1
        left = k
        right = k + L - 1
        
        for j in range(left, right + 1):
            if grid[top][j] is None:
                grid[top][j] = num
                num += 1
                
        for i in range(top + 1, bottom + 1):
            if grid[i][right] is None:
                grid[i][right] = num
                num += 1
                
        if top < bottom:
            for j in range(right - 1, left - 1, -1):
                if grid[bottom][j] is None:
                    grid[bottom][j] = num
                    num += 1
                    
        if left < right:
            for i in range(bottom - 1, top, -1):
                if grid[i][left] is None:
                    grid[i][left] = num
                    num += 1
                    
    for row in grid:
        print(" ".join(str(x) for x in row))

if __name__ == '__main__':
    main()