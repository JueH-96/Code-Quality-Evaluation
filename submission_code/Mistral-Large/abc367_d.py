import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Calculate the cumulative sum of steps
cumulative_sum = [0] * (N + 1)
for i in range(N):
    cumulative_sum[i + 1] = cumulative_sum[i] + A[i]

# Count the number of valid pairs (s, t)
count = 0
for i in range(1, N + 1):
    for j in range(i + 1, N + 1):
        if (cumulative_sum[j] - cumulative_sum[i]) % M == 0:
            count += 1
        if (cumulative_sum[N] - cumulative_sum[j] + cumulative_sum[i]) % M == 0:
            count += 1

print(count)