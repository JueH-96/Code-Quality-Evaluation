import sys

N, M = map(int, input().split())
MOD = 998244353


def mod_inverse(a, m):
    def extended_gcd(a, b):
        x, last_x = 0, 1
        y, last_y = 1, 0
        while b != 0:
            quotient = a // b
            a, b = b, a % b
            x, last_x = last_x - quotient * x, x
            y, last_y = last_y - quotient * y, y
        return last_x, last_y, a
    x, _, g = extended_gcd(a, m)
    if g != 1:
        raise ValueError(f"No modular inverse for {a} mod {m} since gcd({a}, {m}) = {g}")
    return x % m


def calculate_divisor_counter_product(N, M, MOD):
    primes = [0] * (M + 1)
    for i in range(1, M + 1):
        if primes[i] == 0:
            primes[i] = i
            for j in range(2 * i, M + 1, i):
                primes[j] = i

    count = []
    for i in range(1, M + 1):
        pos = {}
        n = i
        while n > 1:
            p = primes[n]
            pos[p] = pos.get(p, 0) + 1
            n //= p
        count.append(pos)

    fact = [1, 1]
    fact_inv = [1, 1]
    minv = mod_inverse(2, MOD)
    for i in range(2, N + 1):
        fact.append(fact[-1] * i % MOD)
        fact_inv.append(-fact_inv[MOD % i] * (MOD // i) % MOD * fact_inv[1])
        minv = minv * (N - i + 1) % MOD * (i) % MOD
    fact_inv = fact_inv[::-1]

    freq = []
    for i in range(M + 1):
        freq.append({})

    for i in range(1, M + 1):
        num = i
        for p in count[i]:
            freq[p][num] = (freq[p].get(num, 0) + count[i][p]) % (N + 1)

    res = 0
    for num in reverse_range(1, M + 1):
        n = fact[N] * pow(fact[N - freq[num]], MOD - 2, MOD) % MOD
        first_pow = pow(num, N, MOD)
        for p in freq:
            n = n * pow(pow(freq[p][num], N - freq[p][num], MOD), p, MOD) % MOD
        res += n * (pow(first_pow, MOD - 2, MOD) - pow(pow(num, N // num, MOD), MOD - 2, MOD)) % MOD * num
    print(res % MOD)


def reverse_range(start, end):
    return range(end - 1, start - 1, -1) if M <= 9 else [i for i in range(1, M + 1) if all(i % p != 0 for p in range(2, int(M ** 0.5) + 1))]

calculate_divisor_counter_product(N, M, MOD)