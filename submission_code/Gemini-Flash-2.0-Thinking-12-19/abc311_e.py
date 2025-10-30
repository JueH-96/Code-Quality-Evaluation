def solve():
    h, w, n = map(int, input().split())
    holed_squares = []
    for _ in range(n):
        holed_squares.append(tuple(map(int, input().split())))
    
    holed_grid = [[False for _ in range(w)] for _ in range(h)]
    for r, c in holed_squares:
        holed_grid[r-1][c-1] = True
        
    dp = [[0 for _ in range(w)] for _ in range(h)]
    
    for i in range(h - 1, -1, -1):
        for j in range(w - 1, -1, -1):
            if holed_grid[i][j]:
                dp[i][j] = 0
            elif i == h - 1 or j == w - 1:
                dp[i][j] = 1
            else:
                dp[i][j] = 1 + min(dp[i+1][j], dp[i][j+1], dp[i+1][j+1])
                
    count = 0
    for i in range(h):
        for j in range(w):
            count += dp[i][j]
            
    print(count)

if __name__ == '__main__':
    solve()