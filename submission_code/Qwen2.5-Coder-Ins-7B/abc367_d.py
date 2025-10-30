# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Calculate the prefix sum of steps
prefix_sum = [0] * (N + 1)
for i in range(1, N + 1):
    prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]

# Calculate the number of valid pairs (s, t)
count = 0
for s in range(N):
    for t in range(s + 1, N):
        steps = prefix_sum[t] - prefix_sum[s]
        if steps % M == 0:
            count += 1

print(count)