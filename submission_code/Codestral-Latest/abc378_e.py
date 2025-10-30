import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Prefix sums and their mod M values
prefix_sum = [0] * (N + 1)
prefix_mod = [0] * (N + 1)

for i in range(1, N + 1):
    prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]
    prefix_mod[i] = prefix_sum[i] % M

# Count the occurrences of each mod value
mod_count = [0] * M
for i in range(1, N + 1):
    mod_count[prefix_mod[i]] += 1

# Calculate the sum of all subarray sums modulo M
total_sum = 0
for i in range(1, N + 1):
    total_sum += prefix_sum[i]

# Calculate the contribution of each mod value
result = 0
for i in range(M):
    result += i * mod_count[i]

# Adjust for the number of subarrays
result *= (N + 1)

# Subtract the total sum of all subarray sums
result -= total_sum

print(result)