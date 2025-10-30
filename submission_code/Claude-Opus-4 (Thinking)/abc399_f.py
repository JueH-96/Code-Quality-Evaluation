# YOUR CODE HERE
N, K = map(int, input().split())
A = list(map(int, input().split()))

MOD = 998244353

# Compute prefix sums
prefix = [0] * (N + 1)
for i in range(N):
    prefix[i + 1] = prefix[i] + A[i]

# Compute the answer
answer = 0
for l in range(1, N + 1):
    for r in range(l, N + 1):
        subarray_sum = prefix[r] - prefix[l - 1]
        answer = (answer + pow(subarray_sum, K, MOD)) % MOD

print(answer)