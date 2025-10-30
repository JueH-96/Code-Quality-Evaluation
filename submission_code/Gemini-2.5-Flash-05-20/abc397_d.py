import sys
import math

# Custom isqrt function to handle large integers and potential floating-point precision issues.
# This is an alternative to math.isqrt (available in Python 3.8+).
# For D up to 1.2 * 10^19, int(D**0.5) might be off by a small integer due to float precision.
# This function checks a small range of integers around the float estimate.
def get_isqrt_and_check_perfect_square(n):
    if n < 0:
        return -1, False # No real square root for negative numbers
    if n == 0:
        return 0, True

    # Initial guess using floating-point square root
    s_float_estimate = n**0.5
    s = int(s_float_estimate)

    # Check integers around the estimate 's'.
    # A range of 5 values (s-2 to s+2) is generally sufficient to compensate for float precision errors.
    for candidate_s in range(max(0, s - 2), s + 3):
        if candidate_s * candidate_s == n:
            return candidate_s, True
    return -1, False # Not a perfect square

def solve():
    N = int(sys.stdin.readline())

    # Iterate 'a' from 1 up to an upper bound.
    # From the derivation a^3 < N, so a < N^(1/3).
    # N_max = 10^18, so N^(1/3) max is 10^6.
    # We add a small constant (e.g., 5) to the upper bound for safety,
    # to account for potential floating-point inaccuracies in N**(1/3).
    a_upper_bound = int(N**(1/3)) + 5
    
    found = False
    for a in range(1, a_upper_bound + 1):
        # 'a' must be a divisor of N
        if N % a == 0:
            b = N // a
            
            # Calculate the discriminant D = 12b - 3a^2
            D = 12 * b - 3 * a * a
            
            # Check if D is a non-negative perfect square
            sqrt_D, is_perfect_square = get_isqrt_and_check_perfect_square(D)
            
            if not is_perfect_square:
                continue # If D is not a perfect square, y is not an integer

            # Calculate the numerator for y: (-3a + sqrt_D)
            # We only consider the '+' case because y must be positive.
            numerator = -3 * a + sqrt_D
            
            # y must be a positive integer
            if numerator <= 0:
                continue # y is not positive
            
            if numerator % 6 == 0:
                y = numerator // 6
                
                # If y is a positive integer, then x = y + a will also be a positive integer
                # (since a >= 1 and y >= 1, x >= 2).
                x = y + a
                print(x, y)
                found = True
                break # A solution is found, print it and exit
    
    if not found:
        print("-1")

solve()