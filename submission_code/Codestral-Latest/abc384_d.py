import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = int(data[1])
A = list(map(int, data[2:]))

# Calculate the prefix sums
prefix_sum = [0] * (2 * N)
for i in range(N):
    prefix_sum[i] = A[i]
    if i > 0:
        prefix_sum[i] += prefix_sum[i - 1]

for i in range(N, 2 * N):
    prefix_sum[i] = prefix_sum[i - N] + A[i % N]

# Use a set to store the prefix sums we have seen so far
seen_prefix_sums = set()
for i in range(2 * N):
    if prefix_sum[i] == S:
        print("Yes")
        break
    if prefix_sum[i] - S in seen_prefix_sums:
        print("Yes")
        break
    seen_prefix_sums.add(prefix_sum[i])
else:
    print("No")