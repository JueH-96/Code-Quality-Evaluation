# YOUR CODE HERE
MOD = 998244353

N, K = map(int, input().split())
A = list(map(int, input().split()))

result = 0

# For each starting position l
for l in range(N):
    current_sum = 0
    # For each ending position r >= l
    for r in range(l, N):
        current_sum = (current_sum + A[r]) % MOD
        # Add (sum of subarray [l,r])^K to result
        result = (result + pow(current_sum, K, MOD)) % MOD

print(result)