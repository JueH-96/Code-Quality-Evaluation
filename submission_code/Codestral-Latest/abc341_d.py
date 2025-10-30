import math

# Read input
N, M, K = map(int, input().split())

# Function to find the K-th smallest number divisible by exactly one of N or M
def find_kth_number(N, M, K):
    # Calculate the least common multiple (LCM) of N and M
    lcm = abs(N * M) // math.gcd(N, M)

    # Initialize pointers for multiples of N and M
    count_n = 0
    count_m = 0
    count_both = 0

    # Binary search to find the K-th number
    low, high = 1, K * max(N, M)
    while low < high:
        mid = (low + high) // 2
        count_n = mid // N
        count_m = mid // M
        count_both = mid // lcm

        # Count numbers divisible by exactly one of N or M
        count_exactly_one = count_n + count_m - 2 * count_both

        if count_exactly_one < K:
            low = mid + 1
        else:
            high = mid

    return low

# Find and print the K-th smallest number
result = find_kth_number(N, M, K)
print(result)