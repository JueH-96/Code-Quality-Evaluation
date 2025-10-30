import sys
MOD = 10**9 + 7

def compute_mobius(max_n):
    mobius = [1] * (max_n + 1)
    is_prime = [True] * (max_n + 1)
    for p in range(2, max_n + 1):
        if is_prime[p]:
            for multiple in range(p, max_n + 1, p):
                is_prime[multiple] = False if multiple != p else is_prime[multiple]
                mobius[multiple] *= -1
            p_square = p * p
            for multiple in range(p_square, max_n + 1, p_square):
                mobius[multiple] = 0
    return mobius

def main():
    nums = list(map(int, sys.stdin.readline().split()))
    max_gcd = max(nums) if nums else 0
    max_gcd = max(max_gcd, 1)
    max_n = max_gcd

    mobius = compute_mobius(max_n)

    c = {}
    for num in nums:
        for g in range(1, num + 1):
            if num % g == 0:
                if g not in c:
                    c[g] = 0
                c[g] += 1

    f = {}
    for g in range(1, max_gcd + 1):
        m = c.get(g, 0)
        if m == 0:
            f[g] = 0
            continue
        mu_sum = 0
        for d in range(1, m + 1):
            if mobius[d] == 0:
                continue
            mu_sum += mobius[d] * (2 ** (m // d) - 1)
        f[g] = mu_sum % MOD

    total = 0
    for g in range(1, max_gcd + 1):
        m = c.get(g, 0)
        if m < 2:
            continue
        sum_g = 0
        for d in range(1, m + 1):
            if mobius[d] == 0:
                continue
            temp = c.get(g * d, 0)
            if temp == 0:
                continue
            sum_g += mobius[d] * (f.get(g * d, 0) * f.get(g * d, 0)) % MOD
        sum_g %= MOD
        total = (total + sum_g) % MOD
    print(total % MOD)

if __name__ == '__main__':
    main()