def main():
    import sys
    import math

    # Read input
    N = int(sys.stdin.readline())

    # Edge case: if N < 1, no such numbers (though per constraints N>=1 anyway)
    if N < 1:
        print(0)
        return

    # We need to count integers ≤ N having exactly 9 divisors.
    # Such integers are of the form p^8 (where p is prime) or p^2 * q^2 (where p, q are distinct primes).

    # Let M = floor(sqrt(N)).
    # For p^2 * q^2 ≤ N => (p*q)^2 ≤ N => p*q ≤ M.
    # We'll sieve primes up to M and then count pairs (p, q) with p < q, p*q ≤ M.

    M = int(math.isqrt(N))

    # Sieve of Eratosthenes to find all primes up to M
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True]*(n+1)
        is_prime[0] = False
        is_prime[1] = False
        # Eliminate even numbers > 2
        for x in range(4, n+1, 2):
            is_prime[x] = False
        limit = int(n**0.5)
        for i in range(3, limit+1, 2):
            if is_prime[i]:
                start = i*i
                step = 2*i
                for multiple in range(start, n+1, step):
                    is_prime[multiple] = False
        return [2] + [x for x in range(3, n+1, 2) if is_prime[x]]

    primes = sieve(M)

    # Count how many p^8 ≤ N
    # In practice, p^8 grows very quickly, so only small p matter.
    count_p8 = 0
    for p in primes:
        # if p^8 <= N
        # (only primes up to about 40 matter for 4e12, but we already have them in 'primes' if M>=p)
        val = p**8
        if val <= N:
            count_p8 += 1
        else:
            break

    # Count how many distinct pairs (p, q), p < q, with p*q ≤ M
    count_pairs = 0
    i = 0
    j = len(primes) - 1
    while i < j:
        if primes[i] * primes[j] <= M:
            # All pairs (i, k) for k in [i+1..j] are valid
            count_pairs += (j - i)
            i += 1
        else:
            j -= 1

    # The total count is those of the form p^8 plus those of the form p^2*q^2
    print(count_p8 + count_pairs)

# Do not forget to call main()
main()