import sys

def solve():
    """
    Reads N from stdin, calculates V_N mod 998244353, and prints the result to stdout.
    """
    # Read the input integer N. Reading as a string first is convenient
    # for getting the number of digits without extra conversions.
    n_str = sys.stdin.readline().strip()
    N = int(n_str)

    # The modulus specified in the problem
    MOD = 998244353

    # Get the number of digits in N
    d = len(n_str)

    # V_N is formed by concatenating N, N times. This can be expressed as
    # a geometric series: V_N = N * (1 + 10^d + ... + (10^d)^(N-1)).
    # We need to compute this modulo MOD.

    # N modulo MOD
    n_mod = N % MOD

    # The common ratio of the series is r = 10^d.
    r = pow(10, d, MOD)

    # Calculate the sum of the geometric series modulo MOD.
    # The formula is (r^N - 1) / (r - 1).
    # A special case exists if the common ratio r is 1 modulo MOD.
    if r == 1:
        # If r = 1, the sum is 1 + 1 + ... + 1 (N times) = N.
        geom_sum_mod = N % MOD
    else:
        # If r != 1, we use the formula. All calculations are modular.

        # Numerator: (r^N - 1) % MOD
        # pow(base, exponent, modulus) is efficient for large exponents.
        r_pow_n = pow(r, N, MOD)
        numerator = (r_pow_n - 1 + MOD) % MOD

        # Denominator: (r - 1) % MOD
        denominator = (r - 1 + MOD) % MOD
        
        # Modular inverse of the denominator. Since MOD is prime, we can use
        # Fermat's Little Theorem: inverse(a) = a^(MOD-2) % MOD.
        inv_denominator = pow(denominator, MOD - 2, MOD)

        # Sum = (numerator * inverse_denominator) % MOD
        geom_sum_mod = (numerator * inv_denominator) % MOD

    # The final result is (N % MOD * geometric_sum % MOD) % MOD
    result = (n_mod * geom_sum_mod) % MOD

    print(result)

solve()