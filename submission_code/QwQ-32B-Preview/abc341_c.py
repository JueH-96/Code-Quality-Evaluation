H, W, N = map(int, input().split())
T = input()
S = [input().strip() for _ in range(H)]

# Convert grid to bitsets
grid = []
for row in S:
    bitset = 0
    for c in row:
        bitset = (bitset << 1) | (1 if c == '.' else 0)
    grid.append(bitset)

# Initialize dp with land cells
dp = grid.copy()

for move in T:
    if move == 'L':
        # Shift left
        new_dp = [0] * H
        for i in range(H):
            new_dp[i] = (dp[i] << 1) & grid[i]
        dp = new_dp
    elif move == 'R':
        # Shift right
        new_dp = [0] * H
        for i in range(H):
            new_dp[i] = (dp[i] >> 1) & grid[i]
        dp = new_dp
    elif move == 'U':
        # Shift up
        new_dp = [0] * H
        for i in range(1, H):
            new_dp[i-1] = dp[i] & grid[i-1]
        dp = new_dp
    elif move == 'D':
        # Shift down
        new_dp = [0] * H
        for i in range(H-1):
            new_dp[i+1] = dp[i] & grid[i+1]
        dp = new_dp

# Count the number of set bits in dp
count = 0
for row in dp:
    count += bin(row).count('1')
print(count)