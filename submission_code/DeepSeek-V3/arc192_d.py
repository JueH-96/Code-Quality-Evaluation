import sys
import math
import bisect

def sieve(max_limit):
    sieve = [True] * (max_limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(max_limit)) + 1):
        if sieve[i]:
            sieve[i*i :: i] = [False] * len(sieve[i*i :: i])
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    return primes

max_A = 10**12
max_prime_needed = 10**6  # because p^2 * q^2 <= 1e12 => p*q <= 1e6, so primes up to 1e6
primes = sieve(max_prime_needed)

# Generate all possible 400 numbers
numbers = set()

# Case 1: p^2 * q^2 where p and q are distinct primes
for i in range(len(primes)):
    p = primes[i]
    p_sq = p * p
    if p_sq > max_A:
        continue
    for j in range(i + 1, len(primes)):
        q = primes[j]
        q_sq = q * q
        product = p_sq * q_sq
        if product > max_A:
            break
        numbers.add(product)
    # Also handle cases where exponents are higher even numbers, but p^2a * q^2b
    # However, the maximum product is minimized when a and b are 1, so higher exponents will lead to smaller numbers only if p and q are very small. But for the problem, the largest number <=A is needed, so higher exponents might yield larger numbers only if the base primes are smaller. However, for the same p and q, p^2 * q^2 is larger than p^4 * q^2 (if 2a=4, 2b=2) only if p^2 < q^2. So to find the maximum possible numbers, we need to include all possible even exponents.
    # So we need to consider all possible even exponents for p and q.
    # But this would be computationally expensive if done naively. Hence, we need a better approach.
    # However, for the problem, the sample includes 36 = 2^2 *3^2, and 400 = 2^4 *5^2. So both cases are covered when considering exponents 2 and 4.
    # So we need to include all numbers of the form p^(2a) * q^(2b) where a and b are >=1.
    # So for each p and q, generate all possible products p^(2a)*q^(2b) <= 1e12.
    a_max = 1
    current_p_pow = p_sq
    while current_p_pow <= max_A:
        # For each a, compute q^(2b) such that product <= max_A
        remaining = max_A // current_p_pow
        if remaining < 4:  # since q >=2, q^2 >=4
            break
        # Now find all q's > p (since i <j)
        for j in range(i + 1, len(primes)):
            q = primes[j]
            q_sq = q * q
            if q_sq > remaining:
                break
            current_product = current_p_pow * q_sq
            numbers.add(current_product)
            # Now try higher powers of q
            current_q_pow = q_sq
            while True:
                current_q_pow *= q * q
                if current_q_pow > remaining:
                    break
                numbers.add(current_p_pow * current_q_pow)
        current_p_pow *= p * p
        a_max += 1

# Case 2: p^2a where a >=2. But this would have only one distinct prime factor, so it's invalid. Hence, no such cases.

# Also, handle cases where one prime is squared to an even exponent >=2 and another to exactly 2.
# But the above nested loops should cover all combinations.

sorted_numbers = sorted(numbers)

Q = int(sys.stdin.readline())
for _ in range(Q):
    A = int(sys.stdin.readline())
    # Find the largest number in sorted_numbers <= A
    idx = bisect.bisect_right(sorted_numbers, A) - 1
    print(sorted_numbers[idx])