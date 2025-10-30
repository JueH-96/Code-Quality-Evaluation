import sys
import math

# Function to calculate the number of positive integers <= x divisible by exactly one of N and M
def count_divisible_by_exactly_one(x, N, M, L):
    """
    Counts positive integers y <= x such that (y%N == 0 and y%M != 0) or (y%N != 0 and y%M == 0).
    This is equivalent to counting y <= x such that (y%N == 0 or y%M == 0) and (y%L != 0).
    Count = |Multiples of N up to x| + |Multiples of M up to x| - 2 * |Multiples of L up to x|
          = floor(x/N) + floor(x/M) - 2 * floor(x/L)
    """
    # Count of multiples of N up to x
    count_N = x // N
    # Count of multiples of M up to x
    count_M = x // M
    # Count of multiples of both N and M (i.e., of L = lcm(N, M)) up to x
    count_L = x // L
    
    # The number of integers <= x divisible by exactly one of N and M is
    # (Number divisible by N) + (Number divisible by M) - 2 * (Number divisible by both N and M)
    return count_N + count_M - 2 * count_L

# Read input
line = sys.stdin.readline().split()
N = int(line[0])
M = int(line[1])
K = int(line[2])

# Calculate LCM(N, M) = (N * M) // GCD(N, M)
# Use math.gcd for efficiency and correctness
common_divisor = math.gcd(N, M)
# Python 3 integers handle large values. N*M can be up to 10^16, which is fine.
# LCM will also be within reasonable bounds relative to N, M.
least_common_multiple = (N * M) // common_divisor

# Binary search for the K-th smallest number
# We are looking for the smallest positive integer `ans` such that count_divisible_by_exactly_one(ans, ...) >= K.
# Search space is positive integers. Minimum possible answer is 1.
# Maximum possible answer estimation:
# The density of such numbers is approximately 1/N + 1/M - 2/LCM.
# The K-th number is roughly K / density.
# In the worst case (N, M large and coprime), density is approx 1/N + 1/M ~ 2/max(N,M).
# K-th number ~ K * max(N,M)/2.
# K=10^10, max(N,M)=10^8. K-th number ~ 10^10 * 10^8 / 2 = 0.5 * 10^18.
# If N=2M, numbers are odd multiples of M. K-th number ~ (2K-1)M ~ 2*10^10*10^8 = 2*10^18.
# A safe upper bound is needed for binary search. 3 * 10^18 seems reasonably safe and covers these cases.
# Any number >= this bound will have a count >> K.
low = 1
high = 3 * 10**18 # A sufficiently large upper bound covering worst-case estimates

ans = high # Initialize ans with a value higher than any potential answer

while low <= high:
    mid = (low + high) // 2
    
    # Calculate the count for 'mid'
    count_mid = count_divisible_by_exactly_one(mid, N, M, least_common_multiple)
    
    if count_mid >= K:
        # mid is a possible answer, try smaller values in the left half
        ans = mid
        high = mid - 1
    else:
        # mid is too small, search in the right half
        low = mid + 1

# Print the result
print(ans)