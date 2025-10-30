import sys
import math

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read N, M, and K from a single line of standard input.
        N, M, K = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handle cases with no input or malformed input.
        return

    # To find numbers divisible by exactly one of N or M, we need their LCM.
    # First, calculate the Greatest Common Divisor (GCD).
    g = math.gcd(N, M)
    # Then, calculate the Least Common Multiple (LCM).
    # lcm(a, b) = (|a * b|) / gcd(a, b).
    # Using (N // g) * M prevents potential intermediate overflow in fixed-size integer languages.
    # In Python, it's not strictly necessary but is good practice.
    lcm = (N // g) * M

    def count_numbers_up_to(x):
        """
        Counts the number of positive integers up to x that are divisible
        by exactly one of N and M.
        
        Formula derived from the Principle of Inclusion-Exclusion:
        Count = (divisible by N) + (divisible by M) - 2 * (divisible by both N and M)
        """
        return (x // N) + (x // M) - 2 * (x // lcm)

    # We perform a binary search on the answer space to find the K-th number.
    # The function count_numbers_up_to(x) is monotonic, which is a prerequisite for binary search.
    
    # Set the search range.
    low = 1
    # The upper bound must be large enough. A rough estimate suggests the answer can be
    # very large (~5e17 for the third sample). 4e18 is a safe upper bound.
    high = 4 * 10**18 
    ans = high

    while low <= high:
        mid = (low + high) // 2
        
        # If mid is 0, we can't perform division. Skip to the next iteration.
        # This check is for robustness; with low starting at 1, mid will be positive.
        if mid == 0:
            low = 1
            continue

        if count_numbers_up_to(mid) >= K:
            # If mid has at least K valid numbers up to it, it might be the answer.
            # We store it and try to find a smaller value.
            ans = mid
            high = mid - 1
        else:
            # If mid has fewer than K valid numbers, it's too small.
            # We need to search in the upper half.
            low = mid + 1

    # The loop terminates when it finds the smallest 'ans' such that
    # count_numbers_up_to(ans) >= K. This is the K-th number.
    print(ans)

# Run the solution.
solve()