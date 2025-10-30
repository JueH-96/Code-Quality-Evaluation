def solve():
    h, w, n = map(int, input().split())
    holed_squares = set()
    for _ in range(n):
        r, c = map(int, input().split())
        holed_squares.add((r, c))
    
    is_holed = [[False for _ in range(w + 1)] for _ in range(h + 1)]
    for r, c in holed_squares:
        is_holed[r][c] = True
        
    dp = [[0 for _ in range(w + 1)] for _ in range(h + 1)]
    total_holeless_squares = 0
    
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            if is_holed[i][j]:
                dp[i][j] = 0
            else:
                if i == 1 or j == 1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            total_holeless_squares += dp[i][j]
            
    print(total_holeless_squares)

if __name__ == '__main__':
    solve()