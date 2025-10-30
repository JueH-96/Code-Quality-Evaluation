import sys
import math

MOD = 998244353

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_str, M_str = sys.stdin.readline().split()
    N = int(N_str)
    M = int(M_str)

    # Precompute primes up to sqrt(N)
    def sieve(n):
        n = int(n)
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i * i, n + 1, i):
                    is_prime[j] = False
        primes = [i for i, val in enumerate(is_prime) if val]
        return primes
    primes = sieve(int(math.isqrt(N)) + 1)

    # Precompute factorial and inverse factorial modulo MOD
    max_fact = M + 60  # Add buffer for exponents up to ~log2(N)
    fact = [1] * (max_fact + 1)
    for i in range(1, max_fact + 1):
        fact[i] = fact[i - 1] * i % MOD
    inv_fact = [1] * (max_fact + 1)
    inv_fact[max_fact] = pow(fact[max_fact], MOD - 2, MOD)
    for i in range(max_fact - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

    def comb(n, k):
        if n < 0 or k < 0 or n < k:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD

    # Memoization dictionaries for S and T
    S_memo = {}
    T_memo = {}

    # Compute sum for S (all f(x))
    def compute_S(n, k, memo):
        if n <= 0:
            return 0
        key = (n, k)
        if key in memo:
            return memo[key]
        # Contribution starts with 1 (x=1)
        res = 1
        if k >= len(primes) or primes[k] > n:
            memo[key] = res
            return res
        p = primes[k]
        res = compute_S(n, k + 1, memo)
        pe = p
        e = 1
        while pe <= n:
            # Compute contribution for p^e
            c = comb(e + M - 1, e)
            q = n // pe
            rec = compute_S(q, k + 1, memo)
            res = (res + c * rec) % MOD
            e += 1
            pe *= p
        memo[key] = res
        return res

    # Compute sum for T (only primes p^e where sigma(p^e) mod 3 !=0)
    def compute_T(n, k, memo):
        if n <= 0:
            return 0
        key = (n, k)
        if key in memo:
            return memo[key]
        res = 1  # Contribution of 1
        if k >= len(primes) or primes[k] > n:
            memo[key] = res
            return res
        p = primes[k]
        # Include all exponents e >=1 where sigma(p^e) mod3 !=0
        res = compute_T(n, k + 1, memo)
        pe = p
        e = 1
        while pe <= n:
            # Determine if sigma(p^e) mod3 ==0
            valid = True
            if p == 3:
                valid = True  # sigma mod3 ==1
            else:
                if p % 3 == 1:
                    if (e + 1) % 3 == 0:
                        valid = False
                elif p % 3 == 2:
                    if (e + 1) % 2 == 0:
                        valid = False
            if valid:
                c = comb(e + M - 1, e)
                q = n // pe
                rec = compute_T(q, k + 1, memo)
                res = (res + c * rec) % MOD
            e += 1
            pe *= p
        memo[key] = res
        return res

    # Compute S and T
    S = compute_S(N, 0, S_memo)
    T = compute_T(N, 0, T_memo)
    ans = (S - T) % MOD
    print(ans)

if __name__ == '__main__':
    main()