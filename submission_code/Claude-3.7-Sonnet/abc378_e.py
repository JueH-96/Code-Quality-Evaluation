# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Precompute prefix sums (mod M)
prefix_sum = [0] * (N+1)
for i in range(1, N+1):
    prefix_sum[i] = (prefix_sum[i-1] + A[i-1]) % M

# Compute the answer
total = 0
for l in range(1, N+1):
    for r in range(l, N+1):
        # The sum of elements from index l to r (mod M)
        sum_lr = (prefix_sum[r] - prefix_sum[l-1]) % M
        total += sum_lr

print(total)