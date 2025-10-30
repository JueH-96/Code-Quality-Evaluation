MOD = 998244353

def count_valid_divisions(N, K, A):
    # Prefix sum array to quickly get the sum of any subarray
    prefix_sum = [0] * (N + 1)
    for i in range(N):
        prefix_sum[i + 1] = prefix_sum[i] + A[i]

    # Dictionary to store the count of subarrays with sum K
    sum_count = {}
    for i in range(1, N + 1):
        for j in range(i):
            subarray_sum = prefix_sum[i] - prefix_sum[j]
            if subarray_sum == K:
                sum_count[(j, i - 1)] = sum_count.get((j, i - 1), 0) + 1

    # Total number of ways to divide the array
    total_ways = 1 << (N - 1)

    # Subtract the number of invalid divisions
    invalid_ways = 0
    for key in sum_count:
        start, end = key
        # Number of ways to include the subarray with sum K
        invalid_ways += 1 << (start + N - 1 - end)

    # Valid divisions are total ways minus invalid ways
    valid_ways = (total_ways - invalid_ways) % MOD
    return valid_ways

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:]))

# Calculate and print the result
result = count_valid_divisions(N, K, A)
print(result)