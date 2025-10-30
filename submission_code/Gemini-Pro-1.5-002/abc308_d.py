# YOUR CODE HERE
def solve():
    h, w = map(int, input().split())
    grid = [input() for _ in range(h)]
    snuke = "snuke"
    
    dp = [[[False] * 5 for _ in range(w)] for _ in range(h)]
    
    if grid[0][0] == snuke[0]:
        dp[0][0][0] = True

    for i in range(h):
        for j in range(w):
            for k in range(5):
                if not dp[i][j][k]:
                    continue
                
                # Move right
                if j + 1 < w and grid[i][j+1] == snuke[(k+1)%5]:
                    dp[i][j+1][(k+1)%5] = True
                
                # Move down
                if i + 1 < h and grid[i+1][j] == snuke[(k+1)%5]:
                    dp[i+1][j][(k+1)%5] = True
                    
    if dp[h-1][w-1][4]:
        print("Yes")
    else:
        print("No")

solve()