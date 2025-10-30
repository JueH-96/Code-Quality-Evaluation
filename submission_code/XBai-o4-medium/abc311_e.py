import sys

def main():
    H, W, N = map(int, sys.stdin.readline().split())
    grid = [[1]*W for _ in range(H)]
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().split())
        grid[a-1][b-1] = 0
    
    dp = [[0]*W for _ in range(H)]
    
    for i in range(H-1, -1, -1):
        for j in range(W-1, -1, -1):
            if grid[i][j] == 0:
                dp[i][j] = 0
            else:
                if i == H-1 or j == W-1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = 1 + min(dp[i+1][j], dp[i][j+1], dp[i+1][j+1])
    
    total = 0
    for row in dp:
        total += sum(row)
    print(total)

if __name__ == "__main__":
    main()