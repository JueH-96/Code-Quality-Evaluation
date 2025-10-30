# YOUR CODE HERE
import sys
from math import gcd

MOD = 998244353

def modinv(x, mod=MOD):
    return pow(x, mod-2, mod)

def solve(N, K):
    # Initial position of the black ball
    pos = 1
    # Probability of the black ball being at the first position
    prob_first = 1
    # Probability of the black ball being at any other position
    prob_other = 0

    for _ in range(K):
        # Probability of not swapping the black ball
        prob_not_swap = (N - 1) * (N - 2) // (N * (N - 1))
        # Probability of swapping the black ball with another ball
        prob_swap = 1 - prob_not_swap

        # Update probabilities
        new_prob_first = prob_first * prob_not_swap + prob_other * prob_swap / (N - 1)
        new_prob_other = prob_first * prob_swap + prob_other * (1 - prob_swap / (N - 1))

        prob_first = new_prob_first
        prob_other = new_prob_other

    # Expected value of the position of the black ball
    expected_value = prob_first * 1 + prob_other * (N + 1) / 2
    # Convert to integer modulo MOD
    expected_value = int(expected_value * modinv(N, MOD)) % MOD

    return expected_value

# Read input
N, K = map(int, sys.stdin.readline().strip().split())

# Solve and print the result
print(solve(N, K))