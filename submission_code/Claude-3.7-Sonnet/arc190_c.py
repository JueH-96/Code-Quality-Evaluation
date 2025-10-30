def main():
    H, W = map(int, input().split())
    grid = [[0] * (W+1) for _ in range(H+1)]
    for h in range(1, H+1):
        values = list(map(int, input().split()))
        for w in range(1, W+1):
            grid[h][w] = values[w-1]
    
    Q, sh, sw = map(int, input().split())
    MOD = 998244353
    
    for _ in range(Q):
        direction, a_i = input().split()
        a_i = int(a_i)
        
        # Move based on direction
        if direction == "L":
            sw -= 1
        elif direction == "R":
            sw += 1
        elif direction == "U":
            sh -= 1
        elif direction == "D":
            sh += 1
        
        # Change the grid value
        grid[sh][sw] = a_i
        
        # Calculate and print the answer
        dp = [[0] * (W+1) for _ in range(H+1)]
        dp[1][1] = grid[1][1]
        
        for w in range(2, W+1):
            dp[1][w] = (dp[1][w-1] * grid[1][w]) % MOD
        
        for h in range(2, H+1):
            dp[h][1] = (dp[h-1][1] * grid[h][1]) % MOD
        
        for h in range(2, H+1):
            for w in range(2, W+1):
                dp[h][w] = (grid[h][w] * ((dp[h-1][w] + dp[h][w-1]) % MOD)) % MOD
        
        print(dp[H][W])

if __name__ == "__main__":
    main()