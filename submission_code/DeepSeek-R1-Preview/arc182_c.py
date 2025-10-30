MOD = 998244353

def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    primes = [2, 3, 5, 7, 11, 13]
    A = {p: 0 for p in primes}
    for x in range(1, M+1):
        for p in primes:
            cnt = 0
            tmp = x
            while tmp % p == 0:
                cnt += 1
                tmp //= p
            A[p] += cnt

    def geometric_sum(a, r, n):
        if r == 1:
            return a * n % MOD
        res = a * (pow(r, n, MOD) - 1) % MOD
        res = res * pow(r - 1, MOD-2, MOD) % MOD
        return res

    def sum_k_Mk(k):
        if k == 0:
            return 0
        M_total = M % MOD
        if M_total == 1:
            return (k * (k + 1) // 2) % MOD
        numerator = (pow(M_total, k, MOD) - 1) * pow(M_total - 1, MOD-2, MOD) % MOD
        numerator = numerator * M_total % MOD
        res = (numerator - k) % MOD
        res = res * pow(M_total - 1, MOD-2, MOD) % MOD
        return res

    T = geometric_sum(1, M, N)
    ans = 1
    for p in primes:
        Ap = A[p] % MOD
        if Ap == 0:
            continue
        sum_k_part = sum_k_Mk(N)
        total = (T + Ap * sum_k_part) % MOD
        ans = ans * total % MOD
    print(ans % MOD)

if __name__ == "__main__":
    main()