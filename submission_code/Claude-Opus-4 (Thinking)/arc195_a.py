n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Use two 1D arrays to represent the current and previous rows
prev = [1] + [0] * m  # prev[j] = dp[i-1][j]
curr = [0] * (m + 1)

for i in range(1, n + 1):
    curr[0] = 1  # dp[i][0] = 1
    for j in range(1, m + 1):
        # Option 1: Don't use A[i-1]
        curr[j] = prev[j]
        
        # Option 2: Use A[i-1] if it matches B[j-1]
        if a[i-1] == b[j-1]:
            curr[j] += prev[j-1]
        
        # Cap at 2 since we only care if it's >= 2
        curr[j] = min(curr[j], 2)
    
    # Swap the arrays
    prev, curr = curr, prev

# The answer is in prev[m] (which was curr[m] after the last iteration)
if prev[m] >= 2:
    print("Yes")
else:
    print("No")