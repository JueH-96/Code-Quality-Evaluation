def solve():
    import sys
    input = sys.stdin.read
    from fractions import Fraction

    MOD = 998244353

    # Read input
    N, M = map(int, input().strip().split())
    total_vertices = N * M

    # E[i] will store the expected number of operations to paint vertex i
    E = [0] * (total_vertices + 1)

    # We will calculate E[i] for i from 1 to N * M
    for i in range(1, total_vertices + 1):
        degree = 1  # Always connected to max(i - N, 0)
        if i > 1:
            degree += 1  # Connected to i - 1
        if i > N:
            degree += 1  # Connected to i - N

        # Calculate E[i]
        sum_neighbors = 0
        if i > 1:
            sum_neighbors += E[i - 1]
        if i > N:
            sum_neighbors += E[i - N]
        if i > 1 and i <= N:
            sum_neighbors += E[0]  # E[0] is 0, but included for completeness

        E[i] = 1 + sum_neighbors / degree

    # Total expected operations
    total_expected = sum(E[1:])  # E[0] is 0, so we sum from 1 to N * M

    # Convert total_expected to a fraction
    total_expected_fraction = Fraction(total_expected).limit_denominator()
    P = total_expected_fraction.numerator
    Q = total_expected_fraction.denominator

    # We need to find R such that R * Q ≡ P (mod MOD)
    # We can find the modular inverse of Q mod MOD
    def mod_inverse(a, mod):
        return pow(a, mod - 2, mod)

    Q_inv = mod_inverse(Q, MOD)
    R = (P * Q_inv) % MOD

    print(R)