import sys
from functools import lru_cache

MOD = 998244353

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def mod_inverse(a, m):
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError(f"No mod inverse exists for {a} under modulo {m}")
    return x % m

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    total_outcomes = 1
    for a in A:
        total_outcomes = (total_outcomes * a) % MOD

    @lru_cache(None)
    def dp(index, current_sum):
        if index == N:
            return 1 if current_sum == 0 else 0
        count = 0
        for x in range(1, A[index] + 1):
            count = (count + dp(index + 1, (current_sum - x) % 10)) % MOD
        return count

    valid_outcomes = dp(0, 10)

    gcd, x, y = extended_gcd(total_outcomes, MOD)
    result = (valid_outcomes * mod_inverse(total_outcomes, MOD)) % MOD
    print(result)

if __name__ == "__main__":
    main()