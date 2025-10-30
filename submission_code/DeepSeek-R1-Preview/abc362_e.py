from collections import defaultdict

mod = 998244353

n = int(input())
A = list(map(int, input().split()))

res = [0] * (n + 2)
res[1] = n  # Each single element is a valid subsequence of length 1

# Initialize dp: each element is a dictionary mapping difference d to another dictionary of lengths and counts
dp = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]

for i in range(n):
    for j in range(i):
        d = A[i] - A[j]
        # Add the subsequence (j, i) of length 2
        dp[i][d][2] += 1
        # Check if j has any entries for this difference and extend them
        if d in dp[j]:
            for l in dp[j][d]:
                new_length = l + 1
                dp[i][d][new_length] += dp[j][d][l]
                dp[i][d][new_length] %= mod  # Apply modulo here to prevent overflow
    # Accumulate the results into the res array
    for d in dp[i]:
        for l in dp[i][d]:
            res[l] += dp[i][d][l]
            res[l] %= mod

# Prepare the output for k=1 to k=n
output = [str(res[k] % mod) for k in range(1, n + 1)]
print(' '.join(output))