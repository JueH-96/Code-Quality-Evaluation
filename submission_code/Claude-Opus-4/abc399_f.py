# YOUR CODE HERE
MOD = 998244353

N, K = map(int, input().split())
A = list(map(int, input().split()))

# Calculate prefix sums
prefix = [0] * (N + 1)
for i in range(N):
    prefix[i + 1] = prefix[i] + A[i]

# Calculate the answer
answer = 0
for l in range(1, N + 1):
    for r in range(l, N + 1):
        # Sum from l to r (1-indexed) is prefix[r] - prefix[l-1]
        subarray_sum = prefix[r] - prefix[l - 1]
        # Add sum^K to answer
        answer = (answer + pow(subarray_sum, K, MOD)) % MOD

print(answer)