import math

# Function to calculate floor(N^(1/k)) using binary search
# This is required because math.pow(N, 1/k) can have precision issues for large N,
# and we need to find the largest integer 'a' such that a^k <= N.
def int_nth_root(val, n):
    if n == 1:
        return val
    if n == 2:
        return math.isqrt(val)

    # For n > 2, use binary search.
    # The upper bound for 'a' such that a^n <= N where N is up to 10^18:
    # If n=3, a_max is 10^6. If n=59, a_max is 2.
    # So 'high' can be set relatively small for n > 2.
    # A safe upper bound for roots is int(val**(1.0/n)) + 2.
    # Python's arbitrary precision integers handle large numbers, so 'mid**n' won't overflow during calculation.
    low = 1
    # Estimate upper bound more carefully to keep binary search efficient,
    # but int(val**(1.0/n)) + 2 is sufficiently robust for Python's large integers.
    # For N=10^18, n=3, int(10^6)+2 = 10^6+2.
    high = int(val**(1.0/n)) + 2 
    
    # Ensure high is not zero or negative for very small val, though low starts at 1
    if high < low:
        high = low # Adjust if N is 0 or 1 and n is large, though N >= 1

    res = 1 # Smallest possible root is 1
    while low <= high:
        mid = (low + high) // 2
        
        # Guard against mid=0 in case low was 0, or mid became 0 (not expected here)
        if mid == 0:
            low = 1
            continue
        
        # Calculate mid^n. Python handles arbitrary large integers, no overflow.
        power = mid**n
        
        # Check if power exceeds val or if it somehow wrapped around to be smaller than mid (unlikely for positive ints)
        # For N=10^18, N_max = 10^18 is exact. Check power > val is sufficient.
        if power > val:
            high = mid - 1
        else: # power <= val
            res = mid
            low = mid + 1
    return res

def solve():
    N = int(input())

    # Step 1: Count all perfect squares <= N.
    # math.isqrt(N) gives floor(sqrt(N)). This correctly counts 1 (as 1^2).
    ans = math.isqrt(N)

    # Step 2: Count non-square perfect powers.
    # These are numbers a^b where 'b' is an odd prime, and the number itself is not a perfect square.
    # We use a set to store these to avoid duplicate counts.
    non_square_powers_seen = set()

    # List of odd prime exponents up to floor(log2(N)) (approx 59 for N=10^18).
    # Even exponents (like 4, 6) are skipped because a^(2k) = (a^k)^2, which is a perfect square.
    # Composite odd exponents (like 9, 15) are skipped because a^(pq) = (a^p)^q,
    # which would be covered by the prime exponent 'q'.
    primes_to_check = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

    for b in primes_to_check:
        # Calculate the maximum base 'a' for the current exponent 'b' such that a^b <= N.
        # 'a' cannot be very large when 'b' is 3 or more (max a is 10^6 for b=3).
        a_limit = int_nth_root(N, b)

        # Iterate 'a' from 2 up to a_limit.
        # We start 'a' from 2 because 1^b = 1 is always a perfect square (1^2) and covered in 'ans'.
        for a in range(2, a_limit + 1):
            val = a**b

            # Check if 'val' is a perfect square.
            # If it is, it's already counted in 'ans' from math.isqrt(N). Skip.
            sqrt_val = math.isqrt(val)
            if sqrt_val * sqrt_val == val:
                continue

            # If 'val' is not a square, check if we've seen this specific value before
            # from a smaller odd prime base (e.g., 2^15 can be (2^3)^5 or (2^5)^3).
            if val in non_square_powers_seen:
                continue

            # If it's a new, distinct non-square perfect power, add it to count and set.
            ans += 1
            non_square_powers_seen.add(val)

    print(ans)

solve()