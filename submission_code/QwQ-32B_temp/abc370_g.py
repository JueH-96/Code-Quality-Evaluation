import sys

MOD = 998244353

def sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n+1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    return primes

def main():
    N, M = map(int, sys.stdin.readline().split())
    if N == 0:
        print(0)
        return

    sqrt_N = int(N**0.5)
    primes = sieve(sqrt_N)

    max_fact = M + 33
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i-1] * i % MOD

    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD-2, MOD)
    for i in range(max_fact -1, -1, -1):
        inv_fact[i] = inv_fact[i+1] * (i+1) % MOD

    def comb(n, k):
        if n < 0 or k < 0 or n < k:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

    def f_g(p, e):
        return comb(e + M -1, M-1)

    def f_bad(p, e):
        if p % 3 == 0:
            return comb(e + M -1, M-1)
        elif p % 3 == 1:
            if e % 3 == 2:
                return 0
            else:
                return comb(e + M -1, M-1)
        else:  # p mod3 ==2
            if e % 2 == 1:
                return 0
            else:
                return comb(e + M -1, M-1)

    def compute_S(N, f):
        memo = {}

        def dfs(K):
            if K == 0:
                return 0
            if K == 1:
                return 1
            if K in memo:
                return memo[K]
            res = 1
            sqrt_K = int(K**0.5)
            for p in primes:
                if p > sqrt_K:
                    break
                max_e = 0
                while (p ** (max_e + 1)) <= K:
                    max_e += 1
                for e in range(1, max_e + 1):
                    pe = p ** e
                    if pe > K:
                        break
                    val = f(p, e)
                    term = val * (dfs(K // pe) - dfs(K // (p**(e+1)))) % MOD
                    res = (res + term) % MOD
            # Check if K is a prime
            is_prime = True
            for p in primes:
                if p * p > K:
                    break
                if K % p == 0:
                    is_prime = False
                    break
            if is_prime and K > 1:
                val = f(K, 1)
                term = val * (dfs(1) - dfs(0)) % MOD
                res = (res + term) % MOD
            memo[K] = res
            return res

        return dfs(N)

    S_total = compute_S(N, f_g)
    S_bad = compute_S(N, f_bad)
    ans = (S_total - S_bad) % MOD
    print(ans)

if __name__ == "__main__":
    main()