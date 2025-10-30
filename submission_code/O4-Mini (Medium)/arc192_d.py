def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    MOD = 998244353

    N = int(input())
    A = list(map(int, input().split()))

    # 1) Build smallest‐prime factor table up to 1000
    MAXA = 1000
    spf = list(range(MAXA+1))
    for i in range(2, int(MAXA**0.5)+1):
        if spf[i] == i:
            for j in range(i*i, MAXA+1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2) Factor each A[i], gather per‐prime exponent sequences
    from collections import defaultdict
    prime_exps = defaultdict(lambda: [0]*(N-1))
    for i in range(N-1):
        x = A[i]
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            prime_exps[p][i] = cnt

    ans = 1

    # 3) For each prime p that actually appears, do the DP
    for p, exps in prime_exps.items():
        # sum of all a_i for this prime
        M = sum(exps)
        if M == 0:
            continue  # contributes factor 1

        # We allow the exponent to range up to 2*M in the worst case
        Emax = 2 * M

        # Precompute p^e mod for e=0..Emax
        powp = [1] * (Emax+1)
        for i in range(1, Emax+1):
            powp[i] = powp[i-1] * p % MOD

        # dp_cur0[e] = total weight arriving at exponent e having NOT seen zero
        # dp_cur1[e] = total weight arriving at e having seen zero at least once
        dp0 = [0]*(Emax+1)
        dp1 = [0]*(Emax+1)

        # Initial: we choose e1 in [0..M], weight p^e1, zero‐seen = (e1==0)
        dp0[:M+1] = powp[:M+1]   # for e>0, zero‐seen = 0
        dp0[0] = 0               # we move that to dp1
        dp1[0] = 1               # weight=1 for e1=0

        cur_max = M  # currently only exponents up to M are nonzero

        # Process transitions for i=1..N-1
        for a in exps:
            # next‐step DP arrays
            nxt0 = [0]*(Emax+1)
            nxt1 = [0]*(Emax+1)
            if a == 0:
                # no change in exponent, only multiply by p^e again
                for e in range(cur_max+1):
                    v0 = dp0[e]
                    if v0:
                        w = v0 * powp[e] % MOD
                        # zero‐seen stays false unless e==0
                        if e == 0:
                            nxt1[0] = (nxt1[0] + w) % MOD
                        else:
                            nxt0[e] = (nxt0[e] + w) % MOD
                    v1 = dp1[e]
                    if v1:
                        w = v1 * powp[e] % MOD
                        # zero‐seen stays true
                        nxt1[e] = (nxt1[e] + w) % MOD
                # cur_max does not change
            else:
                # we branch e -> e+a  or  e -> e-a (if >=0), each time multiply by p^{new_e}
                new_max = cur_max + a
                for e in range(cur_max+1):
                    v0 = dp0[e]
                    v1 = dp1[e]
                    if v0:
                        # branch up
                        eup = e + a
                        wup = v0 * powp[eup] % MOD
                        nxt0[eup] = (nxt0[eup] + wup) % MOD
                        # branch down if valid
                        edn = e - a
                        if edn >= 0:
                            wdn = v0 * powp[edn] % MOD
                            # if edn==0 we see zero now
                            if edn == 0:
                                nxt1[0] = (nxt1[0] + wdn) % MOD
                            else:
                                nxt0[edn] = (nxt0[edn] + wdn) % MOD
                    if v1:
                        # already have seen zero
                        # up
                        eup = e + a
                        wup = v1 * powp[eup] % MOD
                        nxt1[eup] = (nxt1[eup] + wup) % MOD
                        # down
                        edn = e - a
                        if edn >= 0:
                            wdn = v1 * powp[edn] % MOD
                            nxt1[edn] = (nxt1[edn] + wdn) % MOD
                cur_max = new_max

            dp0, dp1 = nxt0, nxt1

        # After step N, our F_p is the sum of dp1[e] for e=0..cur_max
        Fp = sum(dp1[:cur_max+1]) % MOD
        ans = ans * Fp % MOD

    print(ans)


if __name__ == "__main__":
    main()