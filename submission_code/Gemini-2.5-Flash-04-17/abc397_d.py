import sys

def icbrt(n):
    """Integer cube root of non-negative integer n."""
    # Handle negative input; not strictly needed for this problem as N >= 1, so N+1 >= 2.
    if n < 0:
        return -icbrt(-n)
    if n == 0:
        return 0

    # Binary search for the largest integer x such that x**3 <= n.
    # The problem constraint N <= 10**18 means N+1 <= 10**18 + 1.
    # The cube root of 10**18 is 10**6. The cube root of (10**6 + 1)**3 is 10**6 + 1.
    # The maximum value of icbrt(N+1) for N <= 10**18 is icbrt(10**18 + 1) = 10**6.
    # If N is slightly less than (M+1)**3 - 1, N+1 is slightly less than (M+1)**3.
    # icbrt(N+1) can be M. The maximum value of M occurs when (M+1)**3 is just above N+1.
    # Max value of M+1 can be 10**6 + 1 (if N+1 = (10**6+1)**3). So max M = 10**6.
    # A safe upper bound for the binary search result is 10**6 + 1.
    # A safe upper bound for the search range 'high' is 10**6 + 2.
    low = 0
    high = 1000002 # Sufficient upper bound for x in icbrt(n) where n <= 10**18 + 1

    ans = 0
    while low <= high:
        mid = (low + high) // 2
        # mid**3 can be computed directly using Python's arbitrary precision integers.
        # For mid up to 10**6 + 2, mid**3 is approximately 10**18, which fits in Python int.
        mid_cubed = mid ** 3

        if mid_cubed == n:
            return mid
        elif mid_cubed < n:
            ans = mid # mid is a potential answer, search higher
            low = mid + 1
        else: # mid_cubed > n
            high = mid - 1 # mid is too high, search lower

    return ans # ans stores the largest integer whose cube was <= n

def solve():
    # Read input N (positive integer, 1 <= N <= 10**18)
    N = int(sys.stdin.readline())

    # We are looking for positive integers x, y such that x^3 - y^3 = N.
    # Since N > 0, we must have x^3 > y^3, which implies x > y.
    # Since x and y are positive integers, y >= 1.
    # Let k = x - y. k must be a positive integer, k >= 1.
    # Then x = y + k. Substitute this into the equation:
    # (y + k)^3 - y^3 = N
    # Expand the cube: y^3 + 3*y^2*k + 3*y*k^2 + k^3 - y^3 = N
    # Simplify: 3*y^2*k + 3*y*k^2 + k^3 = N

    # We are searching for a positive integer solution y (y >= 1) for some positive integer k (k >= 1).
    # Rearrange the equation as a quadratic in y:
    # (3k) * y^2 + (3k^2) * y + (k^3 - N) = 0

    # For y >= 1, we have 3*y^2*k + 3*y*k^2 + k^3 = k(3y^2 + 3yk + k^2) >= k(3*(1)^2 + 3*(1)*k + k^2) = k(3 + 3k + k^2) = k^3 + 3k^2 + 3k.
    # So, N = 3*y^2*k + 3*y*k^2 + k^3 >= k^3 + 3k^2 + 3k.
    # Thus, k^3 + 3k^2 + 3k <= N.
    # This inequality can be rewritten as (k+1)^3 - 1 <= N, which means (k+1)^3 <= N + 1.
    # Taking the integer cube root of both sides (since k+1 > 0 and N+1 > 0):
    # k + 1 <= icbrt(N + 1)
    # So, the maximum possible integer value for k is icbrt(N + 1) - 1.
    # We can iterate through possible values of k from 1 up to this limit.

    # Calculate the upper limit for k.
    # icbrt(N+1) gives floor((N+1)**(1/3)).
    k_limit = icbrt(N + 1) - 1

    # Iterate through possible values of k starting from 1.
    # The range function is exclusive of the stop value, so range(1, k_limit + 1) iterates from 1 to k_limit.
    # If k_limit is 0 or negative (e.g., for N=1), range(1, <=0) will be empty, which is correct
    # as no solution exists for k >= 1.
    for k in range(1, k_limit + 1):
        # For a fixed k, we check if there exists a positive integer y satisfying the quadratic equation:
        # (3k) * y^2 + (3k^2) * y + (k^3 - N) = 0
        # The discriminant of this quadratic equation is Delta = B^2 - 4AC.
        # Delta = (3k^2)^2 - 4 * (3k) * (k^3 - N)
        # Delta = 9k^4 - 12k * (k^3 - N)
        # Delta = 9k^4 - 12k^4 + 12kN
        Delta = 12 * k * N - 3 * k**4

        # For y to be a real number, the discriminant must be non-negative.
        # If Delta < 0, there are no real solutions for y.
        # Based on the derivation of k_limit (from y >= 1), Delta should be non-negative for k <= k_limit.
        # An explicit check Delta < 0 isn't strictly necessary for correctness given the k_limit derivation,
        # but it can make the loop slightly more efficient if Delta happens to become negative early
        # although this case shouldn't occur for k <= k_limit.
        if Delta < 0:
             continue

        # For y to be an integer, the discriminant must be a perfect square.
        # Calculate the integer square root of Delta.
        # Python's int(float**0.5) works reliably for integers of this magnitude (up to ~10^25).
        m_floor = int(Delta**0.5)

        # Check if m_floor squared is exactly equal to Delta.
        if m_floor * m_floor == Delta:
            # Delta is a perfect square, m = m_floor.
            m = m_floor
            # The integer solutions for y are given by the quadratic formula:
            # y = (-3k^2 +/- m) / (6k)
            # We require y to be a positive integer (y >= 1).
            # For y to be positive, we must use the plus sign for the numerator:
            # y = (-3k^2 + m) / (6k)

            y_num = -3 * k**2 + m

            # For y to be a positive integer, two conditions must be met:
            # 1. The numerator y_num must be positive: y_num > 0. This means -3k^2 + m > 0, or m > 3k^2.
            # 2. The numerator y_num must be divisible by the denominator 6k: y_num % (6k) == 0.

            # Check these two conditions.
            # If y_num is 0, y would be 0, which is not a positive integer. So y_num > 0 is necessary.
            if y_num > 0 and y_num % (6 * k) == 0:
                # If both conditions are met, y = y_num / (6k) is a positive integer.
                y = y_num // (6 * k)
                # We found a pair (x, y) of positive integers.
                x = y + k
                print(x, y)
                return # Found a solution, print it and exit the program.

    # If the loop completes without finding any solution for k in the range [1, k_limit],
    # then no pair of positive integers (x,y) satisfies the equation for the given N.
    print(-1)

# Execute the solve function
solve()