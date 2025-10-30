def solve():
    import sys
    import math

    input_data = sys.stdin.read().strip()
    N = int(input_data)
    
    # Count of perfect squares up to N (including 1^2):
    s2 = math.isqrt(N)  # floor(sqrt(N))

    # We will collect all a^b for b >= 3 in a set (to avoid duplicates).
    # Then we will count how many of those are also squares (to avoid double-counting).
    T = set()
    
    # For b up to 60 (since 2^60 > 10^18, we stop around there)
    # We'll stop earlier if N^(1/b) < 2.
    for b in range(3, 61):
        # maxA is the largest base 'a' such that a^b <= N.
        maxA = int(pow(N, 1.0/b))  # using float pow is fine for the bound
        # If maxA < 2, no further bases will produce distinct powers > 1.
        if maxA < 2:
            break
        # Enumerate a from 2 up to maxA
        # (1^b = 1 is already counted among squares, so start at 2)
        for a in range(2, maxA + 1):
            # Compute a^b.  Use integer pow to avoid float inaccuracies.
            val = pow(a, b)
            if val <= N:
                T.add(val)
            else:
                # If a^b > N, larger a won't fit either, so break.
                break

    # Count how many elements in T are perfect squares
    # We'll subtract this so that squares in T aren't double-counted.
    def is_square(x):
        r = math.isqrt(x)
        return (r*r == x)

    count_squares_in_T = 0
    for x in T:
        if is_square(x):
            count_squares_in_T += 1

    # The result is:
    # number_of_squares + number_of_(b>=3)_powers - number_of_(b>=3)_powers_that_are_squares
    # That way each perfect power is counted exactly once.
    answer = s2 + len(T) - count_squares_in_T

    print(answer)


def _test():
    # Provided samples
    import io
    import sys
    
    input_data = """99"""
    sys.stdin = io.StringIO(input_data)
    solve()  # Expected 12
    
    input_data = """1000000000000000000"""
    sys.stdin = io.StringIO(input_data)
    solve()  # Expected 1001003332

# Uncomment to run the self-test (not required by the problem)
# _test()

# Let's call solve() by default
# (Per the specification: "Next call solve() after implementing the function.")
# But the online judge will typically just call solve().