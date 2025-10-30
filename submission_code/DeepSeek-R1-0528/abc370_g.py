import math
from bisect import bisect_left, bisect_right
import sys

mod = 998244353

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    return primes

def factorial_init(max_val, mod):
    fact = [1] * (max_val + 1)
    inv_fact = [1] * (max_val + 1)
    for i in range(1, max_val + 1):
        fact[i] = fact[i - 1] * i % mod
    inv_fact[max_val] = pow(fact[max_val], mod - 2, mod)
    for i in range(max_val, 0, -1):
        inv_fact[i - 1] = inv_fact[i] * i % mod
    return fact, inv_fact

def nCr(n, r, fact, inv_fact, mod):
    if r < 0 or r > n:
        return 0
    return fact[n] * inv_fact[r] % mod * inv_fact[n - r] % mod

def pi_legendre(n):
    if n < 2:
        return 0
    sqrt_n = int(math.isqrt(n))
    primes = sieve(sqrt_n)
    a = len(primes)
    phi_cache = {}
    def phi(x, a):
        if a == 0:
            return x
        if a == 1:
            return x - (x // 2)
        if (x, a) in phi_cache:
            return phi_cache[(x, a)]
        res = phi(x, a - 1) - phi(x // primes[a - 1], a - 1)
        phi_cache[(x, a)] = res
        return res
    return phi(n, a) + a - 1

def pi_meissel(n):
    if n < 2:
        return 0
    if n < 1e6:
        return pi_legendre(n)
    sqrt_n = int(math.isqrt(n))
    primes = sieve(sqrt_n)
    a = pi_meissel(sqrt_n)
    b = pi_meissel(int(n ** (2/3)))
    c = pi_meissel(int(n ** (1/3)))
    s = b
    for i in range(a, b):
        s -= pi_meissel(n // primes[i]) - i
    return s

def pi_simple(n):
    if n < 2:
        return 0
    if n < 1e6:
        return pi_legendre(n)
    sqrt_n = int(math.isqrt(n))
    primes = sieve(sqrt_n)
    a = len(primes)
    V = [n // i for i in range(1, sqrt_n + 1)]
    V += list(range(V[-1] - 1, 0, -1))
    S = {i: i - 1 for i in V}
    for p in primes:
        for v in V:
            if v < p * p:
                break
            S[v] -= S[v // p] - S[p - 1]
    return S[n]

def pi_2_simple(n):
    if n < 2:
        return 0
    sqrt_n = int(math.isqrt(n))
    primes = sieve(sqrt_n)
    a = len(primes)
    V = [n // i for i in range(1, sqrt_n + 1)]
    V += list(range(V[-1] - 1, 0, -1))
    S = {i: i // 2 for i in V}
    for i in V:
        if i % 3 == 0 or i % 3 == 1:
            S[i] = (i + 2) // 3
        else:
            S[i] = (i + 1) // 3
    for p in primes[1:]:
        if p % 3 != 2:
            continue
        for v in V:
            if v < p * p:
                break
            pv = v // p
            S[v] -= S[pv] - S[p - 1]
    return S[n]

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M_val = int(data[1])
    
    if M_val == 0:
        print(1 if N >= 1 else 0)
        return

    max_val_comb = M_val + 50
    fact, inv_fact = factorial_init(max_val_comb, mod)
    
    def comb(e, M_val):
        return nCr(e + M_val - 1, M_val - 1, fact, inv_fact, mod)
    
    sqrtN = int(math.isqrt(N))
    primes = sieve(sqrtN)
    V = set()
    k = 1
    while k <= sqrtN:
        v1 = N // k
        V.add(v1)
        V.add(k)
        k += 1
    V = sorted(V)
    v2index = {v: idx for idx, v in enumerate(V)}
    
    def min25_sieve(func_type, M_val, primes, V, v2index, N, sqrtN):
        n_primes = len(primes)
        F_next = [0] * len(V)
        if func_type == 'h':
            for idx, v in enumerate(V):
                if v <= sqrtN:
                    count_large_primes = 0
                else:
                    pi_v = pi_simple(v)
                    pi_sqrt = pi_simple(sqrtN)
                    count_large_primes = pi_v - pi_sqrt
                F_next[idx] = count_large_primes * M_val % mod
        else:
            for idx, v in enumerate(V):
                if v <= sqrtN:
                    count_large_primes = 0
                else:
                    pi_v = pi_simple(v)
                    pi_sqrt = pi_simple(sqrtN)
                    pi2_v = pi_2_simple(v)
                    pi2_sqrt = pi_2_simple(sqrtN)
                    count_large_primes = (pi_v - pi_sqrt) - (pi2_v - pi2_sqrt)
                F_next[idx] = count_large_primes * M_val % mod
        
        for i in range(len(primes) - 1, -1, -1):
            p = primes[i]
            F_curr = F_next.copy()
            for j in range(len(V) - 1, -1, -1):
                v = V[j]
                if v < p:
                    break
                power = p
                e = 1
                while power <= v:
                    nv = v // power
                    if nv in v2index:
                        idx_nv = v2index[nv]
                        if func_type == 'h':
                            comb_val = comb(e, M_val)
                        else:
                            if p == 3:
                                comb_val = comb(e, M_val)
                            elif p % 3 == 1:
                                if e % 3 == 2:
                                    comb_val = 0
                                else:
                                    comb_val = comb(e, M_val)
                            else:
                                if e % 2 == 1:
                                    comb_val = 0
                                else:
                                    comb_val = comb(e, M_val)
                        add_val = comb_val * (1 + F_next[idx_nv]) % mod
                        F_curr[j] = (F_curr[j] + add_val) % mod
                    else:
                        add_val = 0
                    next_power = power * p
                    if next_power > v:
                        break
                    power = next_power
                    e += 1
            F_next = F_curr
        
        total = 1
        if N in v2index:
            idxN = v2index[N]
            total = (total + F_next[idxN]) % mod
        return total
    
    T_total = min25_sieve('h', M_val, primes, V, v2index, N, sqrtN)
    G_total = min25_sieve('g', M_val, primes, V, v2index, N, sqrtN)
    ans = (T_total - G_total) % mod
    if ans < 0:
        ans += mod
    print(ans)

if __name__ == '__main__':
    main()