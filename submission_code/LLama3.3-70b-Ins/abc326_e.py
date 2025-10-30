import sys

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
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the expected value
    expected_value = 0
    for i in range(N):
        prob = (i + 1) * mod_inverse(N, 998244353) * mod_inverse(N, 998244353)
        expected_value += prob * A[i]
        for j in range(i + 1, N):
            prob = (i + 1) * mod_inverse(N, 998244353) * mod_inverse(N, 998244353) * (j + 1) * mod_inverse(N, 998244353)
            expected_value += prob * A[j]

    # Calculate the modular inverse of N
    inv_N = mod_inverse(N, 998244353)

    # Calculate the expected value modulo 998244353
    expected_value_mod = 0
    for i in range(N):
        expected_value_mod += A[i] * (i + 1) * inv_N * inv_N
        for j in range(i + 1, N):
            expected_value_mod += A[j] * (i + 1) * inv_N * inv_N * (j + 1) * inv_N

    # Print the result
    print(int(expected_value_mod % 998244353))

if __name__ == "__main__":
    solve()