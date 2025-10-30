# Read the input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Compute the prefix sum
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % M

# Compute the answer
answer = 0
for l in range(1, N + 1):
    for r in range(l, N + 1):
        answer += (prefix_sum[r] - prefix_sum[l - 1]) % M

print(answer)