# YOUR CODE HERE
def count_holeless_squares(H, W, N, holes):
    grid = [[0] * (W + 1) for _ in range(H + 1)]
    
    for a, b in holes:
        grid[a][b] = 1
    
    dp = [[0] * (W + 1) for _ in range(H + 1)]
    
    for i in range(1, H + 1):
        for j in range(1, W + 1):
            if grid[i][j] == 0:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    
    total_holeless_squares = sum(sum(row) for row in dp)
    
    return total_holeless_squares

import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
N = int(data[2])
holes = [(int(data[3 + 2 * i]), int(data[4 + 2 * i])) for i in range(N)]

result = count_holeless_squares(H, W, N, holes)
print(result)