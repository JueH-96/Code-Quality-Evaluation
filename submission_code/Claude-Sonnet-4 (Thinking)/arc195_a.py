n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# dp[j] = number of ways to match first j elements of B (capped at 2)
prev = [0] * (m + 1)
prev[0] = 1  # Base case: empty subsequence

for i in range(n):
    curr = [0] * (m + 1)
    curr[0] = 1  # Base case
    
    for j in range(1, m + 1):
        curr[j] = prev[j]  # Don't use A[i]
        
        if a[i] == b[j-1]:
            curr[j] += prev[j-1]
            # Cap at 2 to avoid overflow
            if curr[j] > 2:
                curr[j] = 2
    
    prev = curr

if prev[m] >= 2:
    print("Yes")
else:
    print("No")