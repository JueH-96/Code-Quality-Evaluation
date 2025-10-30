MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    primes = []
    m = M
    for p in range(2, m+1):
        if m % p == 0:
            primes.append(p)
            while m % p == 0:
                m //= p
    if m > 1:
        primes.append(m)
    
    total = 1
    for p in primes:
        max_e = 0
        cnt = [0] * (M + 1)
        for a in range(1, M+1):
            e = 0
            x = a
            while x % p == 0:
                e += 1
                x //= p
            cnt[e] += 1
        sum_e_plus_1 = [0] * (M + 1)
        for e in range(M + 1):
            sum_e_plus_1[e] = e + 1
        f_p = sum(sum_e_plus_1[e] * cnt[e] for e in range(M + 1))
        f_p_prime = sum(e * cnt[e] for e in range(M + 1))
        r = f_p
        
        if r == 1:
            contrib = N % MOD
        else:
            numerator = (1 - (N + 1) * pow(r, N, MOD) + N * pow(r, N + 1, MOD)) % MOD
            denominator = pow((1 - r) % MOD, 2, MOD)
            inv_denominator = pow(denominator, MOD - 2, MOD)
            S1 = numerator * inv_denominator % MOD
            S2 = r * (pow(r, N, MOD) - 1) % MOD * pow(r - 1, MOD - 2, MOD) % MOD
            contrib = (f_p_prime * S1 % MOD + S2) % MOD
        total = total * contrib % MOD
    print(total)

if __name__ == '__main__':
    main()