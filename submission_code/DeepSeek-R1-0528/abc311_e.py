import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    hole_grid = [[False] * W for _ in range(H)]
    
    index = 3
    for _ in range(N):
        a = int(data[index])
        b = int(data[index+1])
        index += 2
        hole_grid[a-1][b-1] = True
        
    dp = [[0] * W for _ in range(H)]
    
    for i in range(H-1, -1, -1):
        for j in range(W-1, -1, -1):
            if hole_grid[i][j]:
                dp[i][j] = 0
            else:
                down = dp[i+1][j] if i+1 < H else 0
                right = dp[i][j+1] if j+1 < W else 0
                diag = dp[i+1][j+1] if (i+1 < H and j+1 < W) else 0
                dp[i][j] = 1 + min(down, right, diag)
                
    total = 0
    for i in range(H):
        total += sum(dp[i])
        
    print(total)

if __name__ == "__main__":
    main()