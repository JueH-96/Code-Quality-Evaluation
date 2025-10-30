import sys
from itertools import product

MOD = 998244353

def mod_inverse(a, m):
    """Compute the modular inverse of a modulo m"""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        else:
            gcd, x, y = extended_gcd(b % a, a)
            return gcd, y - (b // a) * x, x

    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError("Modular inverse does not exist")
    return x % m

def solve(N, A):
    """Compute the probability that the sum of some dice is 10"""
    total_outcomes = 1
    for a in A:
        total_outcomes = (total_outcomes * a) % MOD

    # Generate all possible outcomes
    outcomes = list(product(*[range(1, a + 1) for a in A]))

    # Count the number of outcomes where the sum of some dice is 10
    count = 0
    for outcome in outcomes:
        for mask in range(1 << N):
            subset_sum = sum(outcome[i] for i in range(N) if (mask & (1 << i)))
            if subset_sum == 10:
                count += 1
                break

    # Compute the probability
    probability = (count * mod_inverse(total_outcomes, MOD)) % MOD
    return probability

def main():
    N, *A = map(int, sys.stdin.readline().split())
    result = solve(N, A)
    print(result)

if __name__ == "__main__":
    main()