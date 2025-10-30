def count_numbers_with_nine_divisors(N):
    import sys
    input = sys.stdin.read
    from math import isqrt

    # Sieve of Eratosthenes to find all primes up to a certain limit
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        p = 2
        while (p * p <= limit):
            if (is_prime[p] == True):
                for i in range(p * p, limit + 1, p):
                    is_prime[i] = False
            p += 1
        prime_list = [p for p in range(2, limit + 1) if is_prime[p]]
        return prime_list

    # Calculate the limit for the sieve
    limit = isqrt(N)
    primes = sieve(limit)

    count = 0

    # Count numbers of the form p^8
    for p in primes:
        if p**8 <= N:
            count += 1
        else:
            break

    # Count numbers of the form p^2 * q
    for i in range(len(primes)):
        p = primes[i]
        p2 = p * p
        if p2 > N:
            break
        for j in range(i + 1, len(primes)):
            q = primes[j]
            if p2 * q <= N:
                count += 1
            else:
                break

    return count

# Reading input
import sys
N = int(sys.stdin.read().strip())

# Calculating and printing the result
result = count_numbers_with_nine_divisors(N)
print(result)