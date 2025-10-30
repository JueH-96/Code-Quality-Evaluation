from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))

prefix_count = defaultdict(int)
prefix_sum = [0] * (N + 1)

for i in range(N):
    prefix_count[A[i]] += 1
    prefix_sum[i + 1] = prefix_sum[i] + prefix_count[A[i]] - prefix_count[A[i] - 1]

print(sum(prefix_sum))