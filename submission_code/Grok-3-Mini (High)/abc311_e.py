import sys
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
N = int(data[index])
index += 1

# Create a grid indicating whether each cell has no hole (True) or is holed (False)
no_hole = [[True] * (W + 1) for _ in range(H + 1)]

# Set the holed cells to False
for _ in range(N):
    a = int(data[index])
    index += 1
    b = int(data[index])
    index += 1
    no_hole[a][b] = False

# Create dp table to store the size of the largest square with bottom-right at (i,j)
dp = [[0] * (W + 1) for _ in range(H + 1)]

# Compute dp and accumulate the sum of all dp values
total = 0
for i in range(1, H + 1):
    for j in range(1, W + 1):
        if not no_hole[i][j]:  # If the cell is holed
            dp[i][j] = 0
        else:
            if i == 1 or j == 1:  # First row or first column
                dp[i][j] = 1
            else:  # General case
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        total += dp[i][j]

# Output the total number of holeless squares
print(total)