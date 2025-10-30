import sys
import math

def solve():
    """
    This function reads an integer N and computes the number of integers x
    between 1 and N that can be expressed as a^b for a>=1, b>=2.
    """
    # It's good practice to read from stdin this way in competitive programming
    # for efficiency, though input() would also work for a single line.
    try:
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        return

    # The problem is to count unique integers x = a^b where 1 <= x <= N, a >= 1, b >= 2.
    # This set of numbers are known as "perfect powers".

    # We split the problem into two main sets:
    # 1. Perfect squares: {k^2 | k >= 1, k^2 <= N}
    # 2. Higher powers: {a^b | a >= 2, b >= 3, a^b <= N}

    # The total number of unique perfect powers is |Squares U Higher Powers|.
    # By the Principle of Inclusion-Exclusion, this is:
    # |Squares| + |Higher Powers| - |Squares ∩ Higher Powers|

    # Part 1: Count the number of perfect squares.
    # The number of integers k >= 1 such that k^2 <= N is floor(sqrt(N)).
    count_squares = int(N**0.5)

    # Part 2: Generate the set of "higher powers".
    # We use a set to automatically handle duplicates (e.g., 64 = 2^6 = 4^3).
    higher_powers = set()

    # The maximum exponent `b` is limited by 2^b <= N.
    # For N <= 10^18, log2(N) is approx. 59.79. So, b_max = 59.
    # A loop up to 60 is safe.
    for b in range(3, 60):
        # Base `a` starts from 2 (a=1 gives 1, which is already in the squares count).
        a = 2
        while True:
            # pow() is efficient for integer exponentiation.
            try:
                p = pow(a, b)
            except OverflowError: # Should not happen in Python, but as a safeguard.
                break

            if p > N:
                # If a^b > N, then (a+1)^b > N, so we are done with this exponent b.
                break
            
            higher_powers.add(p)
            a += 1

        # Optimization: If 2^b > N, the inner loop body was never entered.
        # This means for any larger exponent b', 2^b' will also be > N.
        # So we can stop checking larger exponents.
        if a == 2:
            break

    # Part 3: Count the intersection (higher powers that are also squares).
    count_overlap = 0
    for p in higher_powers:
        # Check if p is a perfect square.
        sqrt_p = int(p**0.5)
        # The precision of float exponentiation is sufficient for this check in Python 3.
        if sqrt_p * sqrt_p == p:
            count_overlap += 1

    # Part 4: Combine the counts using the Principle of Inclusion-Exclusion.
    total_count = count_squares + len(higher_powers) - count_overlap

    print(total_count)

solve()