import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read N and M from a single line of standard input.
        line = sys.stdin.readline()
        if not line:
            return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        # Graceful exit on malformed input.
        return

    # The modulus for the final answer, as specified in the problem.
    MOD = 998244353

    # The problem is equivalent to finding the number of ways to color a cycle graph
    # with N vertices using M colors. This is given by the chromatic polynomial of a
    # cycle graph, P(C_N, M).
    # The formula is: P(C_N, M) = (M-1)^N + (-1)^N * (M-1)
    # All arithmetic is performed modulo MOD.

    # Calculate (M-1). Since M >= 2, M-1 is always non-negative.
    # As M <= 10^6, M-1 is smaller than MOD, so no initial modulo is needed here.
    m_minus_1 = M - 1

    # Calculate the first term: (M-1)^N mod MOD.
    # Python's pow(base, exp, mod) is efficient for this (O(log N)).
    term1 = pow(m_minus_1, N, MOD)

    # Calculate the second term: (-1)^N * (M-1).
    # This depends on the parity of N.
    if N % 2 == 0:
        # If N is even, (-1)^N = 1, so the term is (M-1).
        term2 = m_minus_1
    else:
        # If N is odd, (-1)^N = -1, so the term is -(M-1).
        term2 = -m_minus_1

    # The final answer is the sum of the two terms, modulo MOD.
    # Python's '%' operator correctly handles negative numbers for modular arithmetic
    # (e.g., -5 % 7 == 2), so the result will be in the desired range.
    ans = (term1 + term2) % MOD
    
    # We can add an explicit check to ensure the result is non-negative, which is good practice.
    # However, it's not strictly necessary with Python's % behavior.
    if ans < 0:
        ans += MOD

    # Print the final answer to standard output.
    print(ans)

solve()