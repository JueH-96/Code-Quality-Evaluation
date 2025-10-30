import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Calculate prefix sums
prefix_sum = [0] * (N + 1)
for i in range(N):
    prefix_sum[i + 1] = (prefix_sum[i] + A[i]) % M

# Calculate the frequency of each remainder
remainder_count = [0] * M
for s in prefix_sum:
    remainder_count[s] += 1

# Calculate the result
result = 0
for count in remainder_count:
    result += count * (count - 1) // 2

# Add the contribution of single elements
for i in range(N):
    result += A[i] % M

print(result)