def count_holeless_squares(H, W, holes):
    if not holes:
        # Formula for the number of holeless squares in a grid with no holed squares
        total = 0
        min_dim = min(H, W)
        
        for k in range(1, min_dim + 1):
            total += (H - k + 1) * (W - k + 1)
        
        return total
    
    # Convert holes to 0-indexed for easier implementation
    holes_0indexed = [(a-1, b-1) for a, b in holes]
    holed_positions = set(holes_0indexed)
    
    # DP table for the size of the largest holeless square ending at (i, j)
    dp = [[0 for _ in range(W)] for _ in range(H)]
    
    count = 0
    
    for i in range(H):
        for j in range(W):
            if (i, j) in holed_positions:  # If position (i, j) is holed
                dp[i][j] = 0
            else:
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
                count += dp[i][j]
    
    return count

H, W, N = map(int, input().split())
holes = []
for _ in range(N):
    a, b = map(int, input().split())
    holes.append((a, b))

result = count_holeless_squares(H, W, holes)
print(result)