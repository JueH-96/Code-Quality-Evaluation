def main():
    import sys
    import math

    input_data = sys.stdin.read().strip()
    N = int(input_data)

    # We are looking for numbers <= N that have exactly 9 positive divisors.
    # Such numbers factor in one of two ways:
    #
    # 1) A perfect 8th power of a prime: p^8
    #    The divisor count is (8 + 1) = 9.
    #
    # 2) The product of squares of two distinct primes: p^2 * q^2
    #    The divisor count is (2 + 1)*(2 + 1) = 9.
    #
    # We'll count both kinds up to N.

    # ----------------------------------------------------------------------
    # 1) Count all primes p where p^8 <= N.
    #    We'll find all primes up to floor(N^(1/8)) and check p^8 <= N.
    # ----------------------------------------------------------------------

    # We define a function to find primes up to limit using a sieve.
    def sieve_primes_up_to(limit):
        """Return a list of primes <= limit."""
        if limit < 2:
            return []
        # Sieve of Eratosthenes
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i in range(2, limit + 1) if sieve[i]]

    # Compute integer 8th root of N to bound primes for p^8 <= N.
    # We'll pick a candidate by taking int(N**(1/8)) with some adjustments.
    p8 = int(N**(1/8))
    while (p8+1)**8 <= N:
        p8 += 1
    while p8**8 > N and p8 > 0:
        p8 -= 1

    # Sieve primes up to p8
    primes_for_8th = sieve_primes_up_to(p8)
    count_8th_powers = 0
    for p in primes_for_8th:
        if p**8 <= N:
            count_8th_powers += 1
        else:
            break

    # ----------------------------------------------------------------------
    # 2) Count numbers of the form p^2 * q^2 with p < q, p and q primes, <= N.
    #    We need p^2 * q^2 <= N => p*q <= sqrt(N).
    # ----------------------------------------------------------------------

    # We'll find all primes up to floor(sqrt(N)). Then use a two-pointer (or
    # similar) approach to count pairs (p, q) with p < q and p*q <= sqrtN.

    sqrtN = int(math.isqrt(N))  # floor of sqrt(N)

    primes_up_to_sqrtN = sieve_primes_up_to(sqrtN)

    count_p2_q2 = 0
    # Two-pointer approach: we want to count all pairs (p, q) with p < q
    # and p*q <= sqrtN.
    # List of primes: p0 < p1 < p2 < ... < p_last
    # We'll iterate over p in ascending order and find how many q > p satisfy p*q <= sqrtN.

    j = len(primes_up_to_sqrtN) - 1
    for i, p in enumerate(primes_up_to_sqrtN):
        # Move j down while p * primes_up_to_sqrtN[j] > sqrtN
        while j > i and p * primes_up_to_sqrtN[j] > sqrtN:
            j -= 1
        # Now primes_up_to_sqrtN[j] is the largest prime s.t. p*q <= sqrtN
        # The valid q's for this p are from i+1 to j (ensuring q>p => index>i).
        if j > i:
            count_p2_q2 += (j - i)

    # Total count = count_8th_powers + count_p2_q2
    ans = count_8th_powers + count_p2_q2

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()