# YOUR CODE HERE
import math
import sys

# Function to compute greatest common divisor (GCD) using Euclidean Algorithm
def gcd(a, b):
    """Computes the greatest common divisor of non-negative integers a and b."""
    while b:
        a, b = b, a % b
    # The GCD is the value of a when b becomes 0
    return a

# Function to compute least common multiple (LCM)
def lcm(a, b):
    """Computes the least common multiple of positive integers a and b."""
    # Use the formula lcm(a, b) = (|a * b|) / gcd(a, b)
    # Since N and M are positive integers, a and b will be positive.
    if a == 0 or b == 0:
        # This case should not occur based on problem constraints (N, M >= 1)
        # Included for robustness, returning 0 might be conventional.
        return 0 
    
    # Python's arbitrary precision integers handle potentially large intermediate product (a * b)
    # Integer division `//` ensures the result is an integer.
    common_divisor = gcd(a, b)
    # The calculation `(a // common_divisor) * b` can sometimes help avoid overflow 
    # in fixed-size integer types, but Python handles large integers automatically.
    # So, `(a * b) // common_divisor` is fine and perhaps more standard.
    return (a * b) // common_divisor


# Function to count the number of positive integers less than or equal to x
# that are divisible by exactly one of N and M.
def count_valid(x, N, M, L):
    """Counts the number of positive integers <= x divisible by exactly one of N and M.
    L is the least common multiple of N and M."""
    
    # If x is non-positive, there are no positive integers <= x.
    if x <= 0:
        return 0
    
    # Count multiples of N up to x
    count_N = x // N
    # Count multiples of M up to x
    count_M = x // M
    
    # Count multiples of L = lcm(N, M) up to x
    # These are the numbers divisible by both N and M.
    # Since N, M >= 1, their LCM L will also be >= 1, so division by L is safe.
    count_L = x // L
    
    # The principle of inclusion-exclusion applied for symmetric difference:
    # |A union B| = |A| + |B| - |A intersect B|
    # |A delta B| = |A union B| - |A intersect B|
    # |A delta B| = (|A| + |B| - |A intersect B|) - |A intersect B|
    # |A delta B| = |A| + |B| - 2 * |A intersect B|
    # Here A is the set of multiples of N, B is the set of multiples of M.
    # A intersect B is the set of multiples of L = lcm(N, M).
    # So, the count is count_N + count_M - 2 * count_L.
    return count_N + count_M - 2 * count_L

# Main part of the script implementing the solution strategy
def solve():
    # Read input N, M, K from standard input
    # Input is guaranteed to be three positive integers.
    line = sys.stdin.readline().split()
    N = int(line[0])
    M = int(line[1])
    K = int(line[2])

    # Calculate the least common multiple of N and M
    # This is needed for the count_valid function.
    L = lcm(N, M)

    # We need to find the K-th smallest positive integer `X` such that `X` is divisible
    # by exactly one of N and M.
    # We can use binary search on the answer `X`.
    # The function `count_valid(x, N, M, L)` is monotonically non-decreasing with `x`.
    # We are looking for the smallest `X` such that `count_valid(X, N, M, L) >= K`.

    # Set the search range for binary search.
    # The lower bound can be 1 (smallest positive integer).
    low = 1
    # The upper bound needs to be sufficiently large.
    # Analysis shows that the K-th number could be up to approx K * max(N, M) / density.
    # Maximum K is 10^10, max(N, M) is 10^8. The maximum answer could be around 10^18.
    # An upper bound of 2 * 10^18 is safe.
    high = 2 * 10**18 
    # Initialize answer variable. It will store the smallest `mid` found so far that satisfies `count_valid(mid) >= K`.
    ans = high # Defaulting to the high end ensures `ans` is updated if a valid `mid` is found.

    while low <= high:
        # Calculate midpoint. Using `low + (high - low) // 2` is safer against potential overflow
        # in languages with fixed-size integers, although Python handles arbitrary precision integers.
        mid = low + (high - low) // 2
        
        # Check if `mid` could be the K-th number or greater.
        # Evaluate the count function at the midpoint.
        count = count_valid(mid, N, M, L)
        
        # Adjust the binary search range based on the comparison of `count` with `K`.
        if count >= K:
            # If `count_valid(mid)` is at least K, then `mid` is a potential candidate for the K-th number.
            # It might be the K-th number, or some number larger than the K-th.
            # We store `mid` as the current best answer (smallest found so far)
            # and try to find an even smaller valid number by searching in the lower half: [low, mid - 1].
            ans = mid
            high = mid - 1
        else: # count < K
            # If `count_valid(mid)` is less than K, then `mid` is too small.
            # The K-th number must be larger than `mid`.
            # We need to search in the upper half: [mid + 1, high].
            low = mid + 1

    # After the binary search loop terminates (`low > high`),
    # `ans` holds the smallest integer `X` such that `count_valid(X) >= K`.
    # This is the K-th smallest positive integer divisible by exactly one of N and M.
    print(ans)

# Execute the solve function when the script is run
solve()