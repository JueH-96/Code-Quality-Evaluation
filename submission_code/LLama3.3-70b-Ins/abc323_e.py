import sys
from functools import reduce
from operator import mul

MOD = 998244353

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def read_int():
    return int(sys.stdin.readline())

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        gcd, x, y = extended_gcd(b % a, a)
        return gcd, y - (b // a) * x, x

def mod_inverse(a, m):
    gcd, x, y = extended_gcd(a, m)
    if gcd != 1:
        return None
    else:
        return x % m

def solve():
    N, X = read_ints()
    T = read_ints()

    # Calculate the probability of song 1 being played at time X + 0.5
    # We can use the fact that the probability of song 1 being played at time X + 0.5
    # is equal to the probability of song 1 being played at time X + 0.5 in the first N songs
    # plus the probability of song 1 being played at time X + 0.5 in the remaining songs

    # Calculate the probability of song 1 being played at time X + 0.5 in the first N songs
    prob_first_N = 0
    for i in range(N):
        if T[i] == X + 1:
            prob_first_N += 1 / N

    # Calculate the probability of song 1 being played at time X + 0.5 in the remaining songs
    prob_remaining = 0
    for i in range(1, X + 1):
        for j in range(N):
            if T[j] == i:
                prob_remaining += (1 / N) * (1 / N) ** (X - i)

    # Calculate the total probability
    total_prob = (prob_first_N + prob_remaining) % MOD

    # Calculate the modular inverse of N
    inv_N = mod_inverse(N, MOD)

    # Calculate the final probability
    final_prob = (total_prob * inv_N) % MOD

    print(int(final_prob))

if __name__ == "__main__":
    solve()