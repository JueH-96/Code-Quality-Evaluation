# YOUR CODE HERE
import sys

# Memoization dictionary
# Key: (current_index, tight_constraint, max_allowed_digit_exclusive)
# current_index: The 0-based index of the digit being filled in the original number string S.
# tight_constraint: Boolean, True if the digits being filled are restricted by the original number S.
# max_allowed_digit_exclusive: The upper bound (exclusive) for the digits allowed at indices > 0.
#                              This value is fixed by the first digit of the number being built.
memo = {}
S_global = "" # Store the string representation of N globally for the DP function

def solve_from_second_digit(index, tight, max_allowed_digit_exclusive):
    """
    Counts valid suffixes of a number, starting from the digit at `index`.
    A suffix is valid if all its digits are strictly less than `max_allowed_digit_exclusive`.
    The count is restricted by the `tight` constraint relative to S_global[index:].
    """
    D = len(S_global)

    # Base case: Successfully filled all subsequent digits (from index 1 to D-1)
    if index == D:
        return 1

    # Check memoization
    if (index, tight, max_allowed_digit_exclusive) in memo:
        return memo[(index, tight, max_allowed_digit_exclusive)]

    # Determine the upper bound for the current digit (at S_global[index])
    # The digit must be less than max_allowed_digit_exclusive.
    # The digit must also be <= the digit in S_global[index] if tight constraint is active.
    limit_digit = int(S_global[index]) if tight else 9

    # The current digit `d` must satisfy: 0 <= d < max_allowed_digit_exclusive
    # Also, if tight, d must satisfy: d <= limit_digit.
    # So, the actual upper bound for `d` is min(max_allowed_digit_exclusive - 1, limit_digit).
    upper = min(max_allowed_digit_exclusive - 1, limit_digit)

    ans = 0
    # Iterate through possible digits for the current position (index).
    # Allowed digits are from 0 up to `upper`.
    for d in range(upper + 1):
        new_tight = tight and (d == limit_digit)
        ans += solve_from_second_digit(index + 1, new_tight, max_allowed_digit_exclusive)

    # Store result in memo and return
    memo[(index, tight, max_allowed_digit_exclusive)] = ans
    return ans

def count(N_val):
    """
    Counts the number of Snake numbers X such that 10 <= X <= N_val.
    """
    # Snake numbers are >= 10
    if N_val < 10:
        return 0

    global S_global, memo
    S_global = str(N_val)
    D = len(S_global)
    total_count = 0

    # 1. Count Snake numbers with fewer than D digits.
    # A d-digit number (d >= 2) is Snake if the first digit k (1..9)
    # is strictly greater than the remaining d-1 digits (0..k-1).
    # Number of such numbers for fixed k is k^(d-1).
    for d in range(2, D): # Lengths from 2 up to D-1
        for k in range(1, 10): # First digit k (1 to 9)
            # All k^(d-1) numbers of length d starting with k and having
            # subsequent digits < k are valid Snake numbers.
            total_count += k**(d-1)

    # 2. Count Snake numbers with exactly D digits that are <= N_val.
    # Let the number be X = x_1 x_2 ... x_D.
    # x_1 must be in [1, 9]. x_i < x_1 for i > 1. And X <= N_val.
    s1 = int(S_global[0]) # The first digit of N_val

    # Iterate through possible first digits k for X (1 <= k <= 9).
    # For X to be <= N_val (and have same number of digits D), the first digit k must be <= s1.
    # If k > s1, any number X starting with k will be > N_global.
    for k in range(1, s1 + 1):
        if k < s1:
            # If k < s1, any choice of remaining D-1 digits < k results in X < s1 0...0 <= N_val.
            # Number of choices for D-1 digits (each from 0 to k-1) is k^(D-1).
            total_count += k**(D-1)
        elif k == s1: # This branch handles the case k = s1
            # If k == s1, the first digit is fixed to s1.
            # We need to choose remaining D-1 digits x_2...x_D
            # such that 0 <= x_i < k=s1 AND s1 x_2...x_D <= S_global.
            # This is equivalent to counting sequences x_2...x_D where 0 <= x_i < s1
            # and the number formed by x_2...x_D is <= the number formed by S_global[1:].
            # Use DP `solve_from_second_digit` starting from index 1 of S_global.
            # The tight constraint is initially True as we are restricted by S_global[1:].
            # The maximum allowed digit (exclusive) for positions 1 to D-1 is s1.
            memo = {} # Reset memoization for this DP call
            total_count += solve_from_second_digit(1, True, s1)

    return total_count

# Read input L and R
L, R = map(int, sys.stdin.readline().split())

# The number of Snake numbers between L and R (inclusive) is count(R) - count(L-1).
result = count(R) - count(L - 1)

# Print the result
print(result)