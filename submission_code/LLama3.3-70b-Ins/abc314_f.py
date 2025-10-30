import sys
from collections import defaultdict

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

def solve():
    """Solve the problem"""
    N = int(sys.stdin.readline())
    matches = []
    for _ in range(N - 1):
        p, q = map(int, sys.stdin.readline().split())
        matches.append((p, q))

    # Initialize the expected values
    expected = [0] * (N + 1)

    # Process the matches in reverse order
    for i in range(N - 2, -1, -1):
        p, q = matches[i]
        # Compute the probability of team p winning
        prob_p = 1 / (1 + 1)
        # Update the expected values
        expected[p] = (expected[p] + prob_p) % MOD
        expected[q] = (expected[q] + (1 - prob_p)) % MOD

    # Print the expected values
    print(' '.join(map(str, expected[1:])))

if __name__ == "__main__":
    solve()