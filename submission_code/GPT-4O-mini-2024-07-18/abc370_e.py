def count_divisions(N, K, A):
    MOD = 998244353
    
    # Prefix sum array to store cumulative sums
    prefix_sum = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_sum[i] = prefix_sum[i - 1] + A[i - 1]
    
    # Dictionary to count occurrences of prefix sums
    prefix_count = {}
    for i in range(N + 1):
        current_sum = prefix_sum[i]
        if current_sum in prefix_count:
            prefix_count[current_sum] += 1
        else:
            prefix_count[current_sum] = 1
    
    # Total ways to divide the array
    total_ways = pow(2, N - 1, MOD)
    
    # Count the invalid divisions
    invalid_count = 0
    for i in range(1, N + 1):
        target_sum = prefix_sum[i] - K
        if target_sum in prefix_count:
            invalid_count += prefix_count[target_sum]
    
    # Valid divisions are total ways minus invalid ones
    valid_count = (total_ways - invalid_count) % MOD
    
    return valid_count

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
A = list(map(int, data[2:N + 2]))

# Get the result and print it
result = count_divisions(N, K, A)
print(result)