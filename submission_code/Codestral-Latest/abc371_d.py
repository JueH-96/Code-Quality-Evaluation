import sys
from bisect import bisect_left, bisect_right

input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
X = list(map(int, data[index:index + N]))
index += N
P = list(map(int, data[index:index + N]))
index += N
Q = int(data[index])
index += 1

prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = prefix_sum[i] + P[i]

results = []
for _ in range(Q):
    L = int(data[index])
    R = int(data[index + 1])
    index += 2

    left_index = bisect_left(X, L)
    right_index = bisect_right(X, R)

    total_villagers = prefix_sum[right_index] - prefix_sum[left_index]
    results.append(total_villagers)

for result in results:
    print(result)