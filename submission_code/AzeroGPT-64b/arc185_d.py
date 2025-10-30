def fast_mod_exp(base, exponent, modulus):
    result = 1
    base = base % modulus

    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus

    return result

def mod_inverse(n, modulus):
    return fast_mod_exp(n, modulus - 2, modulus)

modulus = 998244353

def extended_gcd(a, b):
    old_r, r = a, b
    old_s, s = 1, 0
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    return old_r, old_s, (old_r - old_s * a) // b if b else 0

def mod_inverse2(a, m):
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError("Modular inverse not defined")
    else:
        return x % m

from math import comb

def solve(N, M):
    T = N * M + 1
    ans_row, ans_col = N * M + 2, N + 2
    answer = pow(2, T, modulus) - 1 + pow(2, T, modulus) * (N + M)

    for row in range(2, N + 2):
        if (T + 2) % row == 0:
            answer -= pow(2, T + 1, modulus) * comb(T + 2, row)
        else:
            for col in range(2, N + 2):
                if row * col > T + 2:
                    break
                if (T + 2) % (row * col) == 0:
                    answer -= pow(2, T + 1, modulus) * comb(T + 2, row * col)

    return (answer * mod_inverse(2, modulus) + 1) % modulus - 1

N, M = map(int, input().split())
print(solve(N, M))