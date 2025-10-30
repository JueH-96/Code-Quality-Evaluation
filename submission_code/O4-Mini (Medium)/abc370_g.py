import threading
def main():
    import sys
    sys.setrecursionlimit(1000000)
    mod = 998244353
    N, M = map(int, sys.stdin.readline().split())

    # Precompute factorials/inverses up to M+40 for combinations
    maxE = 40  # since 2^40 > 1e10
    LIM = M + maxE + 5
    fact = [1] * (LIM)
    invf = [1] * (LIM)
    for i in range(1, LIM):
        fact[i] = fact[i-1] * i % mod
    invf[LIM-1] = pow(fact[LIM-1], mod-2, mod)
    for i in range(LIM-1, 0, -1):
        invf[i-1] = invf[i] * i % mod
    # combination nCk mod
    def comb(n, k):
        if n<0 or k<0 or k>n: return 0
        return fact[n] * invf[k] % mod * invf[n-k] % mod

    # Precompute f_e = C(e+M-1, M-1) for e up to maxE
    f_e = [comb(e+M-1, M-1) for e in range(maxE+1)]
    # f_bad for bad numbers: two versions depending on prime class
    # For p % 3 == 1: exponents e mod3==2 are forbidden (f=0)
    # For p % 3 == 2: odd exponents forbidden

    import math
    # Sieve primes up to N^(1/3)+10
    C = int(N**(1/3)) + 5
    sieve = [True]*(C+1)
    primes = []
    for i in range(2, C+1):
        if sieve[i]:
            primes.append(i)
            for j in range(i*i, C+1, i):
                sieve[j] = False
    # Min_25 sieve supporting two multiplicative functions f and f_bad
    # We'll build the list of all v = N//i distinct, and also i itself
    V = []
    i = 1
    while i <= N:
        q = N//i
        V.append(q)
        i = N//q + 1
    W = sorted(set(V))
    idx = {w: k for k,w in enumerate(W)}
    sz = len(W)
    # dp_total[k] = sum_{n<=W[k]} f(n), dp_bad similarly
    dp_total = [0]*sz
    dp_bad = [0]*sz
    for k,w in enumerate(W):
        # initialize to sum_{n=1..w} f(1-based) if f were 1 for all n
        # but f(1)=1 and for primes we'd subtract later
        dp_total[k] = w % mod
        dp_bad[k]   = w % mod

    # Now sieve out primes contribution in Min_25 style
    # We iterate primes p, for each w>=p*p, dp[w] -= sum_{e>=1, p^e <= w} f(p^e)*dp[w//p^e] for large primes
    # but also handle small prime part
    # We also need prefix sums of f(p) over primes for w < p*p
    # So prefix_total[i] = sum_{primes[j]<primes[i]} f(primes[j])
    prefix_tot = [0]*(len(primes)+1)
    prefix_bad = [0]*(len(primes)+1)
    for i,p in enumerate(primes):
        prefix_tot[i+1] = (prefix_tot[i] + f_e[1]) % mod  # f(p)=f_e[1]
        # bad version
        if p == 3:
            prefix_bad[i+1] = (prefix_bad[i] + f_e[1]) % mod
        elif p%3==1:
            # e=1 mod3, ok
            prefix_bad[i+1] = (prefix_bad[i] + f_e[1]) % mod
        else:
            # p%3==2, e=1 odd -> bad f=0
            prefix_bad[i+1] = prefix_bad[i]

    # Main Min_25 loops
    for i,p in enumerate(primes):
        p2 = p*p
        for k,w in enumerate(W):
            if w < p2: break
            val_total = 0
            val_bad   = 0
            # accumulate for e>=1
            pk = p
            for e in range(1, maxE+1):
                if pk > w: break
                fe = f_e[e]
                # bad f?
                if p == 3:
                    fe_bad = fe
                elif p%3==1:
                    fe_bad = fe if (e%3)!=2 else 0
                else:
                    fe_bad = fe if (e%2)==0 else 0
                # pick dp at w//p^e
                j = idx[w//pk]
                val_total = (val_total + fe * dp_total[j]) % mod
                val_bad   = (val_bad   + fe_bad * dp_bad[j]) % mod
                pk *= p
            # subtract from dp
            dp_total[k] = (dp_total[k] - val_total) % mod
            dp_bad[k]   = (dp_bad[k]   - val_bad)   % mod
        # update prefix sums for w < p^2
        # nothing needed here as we use prefix for small w in final stage

    # After sieve, dp_total[k] = sum_{n<=W[k]} f(n) including 1
    # dp_bad similarly. Our answer is dp_total for W[k] where W[k] == N minus dp_bad
    ans = (dp_total[idx[N]] - dp_bad[idx[N]]) % mod
    print(ans)

if __name__ == "__main__":
    main()