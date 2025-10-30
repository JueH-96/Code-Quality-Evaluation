n = int(input())
if n == 0:
    print(0)
    exit()

# Initialize the distance matrix
d = [[0] * n for _ in range(n)]
for i in range(n - 1):
    row = list(map(int, input().split()))
    for j in range(i + 1, n):
        idx = j - (i + 1)
        d[i][j] = row[idx]
        d[j][i] = d[i][j]

max_mask = 1 << n
dp = [-float('inf')] * max_mask
initial_mask = (1 << n) - 1
dp[initial_mask] = 0

# Sort masks in descending order based on the number of set bits
masks = sorted(range(max_mask), key=lambda x: bin(x).count('1'), reverse=True)

for mask in masks:
    if dp[mask] == -float('inf'):
        continue
    # Generate all possible pairs (i, j) in the current mask
    for i in range(n):
        if not (mask & (1 << i)):
            continue
        for j in range(i + 1, n):
            if mask & (1 << j):
                new_mask = mask & ~((1 << i) | (1 << j))
                if dp[new_mask] < dp[mask] + d[i][j]:
                    dp[new_mask] = dp[mask] + d[i][j]

# The answer is the maximum value in the dp array
print(max(dp))