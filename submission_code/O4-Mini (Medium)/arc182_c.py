import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    MOD = 998244353

    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])

    # list of primes up to M
    primes = []
    for p in [2,3,5,7,11,13]:
        if p <= M:
            primes.append(p)
    r = len(primes)

    # exponents of each x in [1..M] for each prime
    e = [[0]*r for _ in range(M+1)]
    for x in range(1, M+1):
        v = x
        for i,p in enumerate(primes):
            cnt = 0
            while v % p == 0:
                v //= p
                cnt += 1
            e[x][i] = cnt

    # precompute c_B for non-empty subsets B (bitmask 1..(1<<r)-1)
    # where c_B = sum_{x=1..M} prod_{i in B} e[x][i]
    maxmask = 1<<r
    cB = [0]*maxmask
    for mask in range(1, maxmask):
        s = 0
        for x in range(1, M+1):
            prod = 1
            # if any exponent zero for a prime in B, prod=0
            m = mask
            ok = True
            while m:
                i = (m & -m).bit_length() - 1
                m &= m-1
                ei = e[x][i]
                if ei == 0:
                    ok = False
                    break
                prod = (prod * ei) % MOD
            if ok:
                s = (s + prod) % MOD
        cB[mask] = s

    # small binomial comb[n][k] for n,k<=7
    maxn = r+1  # <=7
    comb_small = [[0]*(maxn+1) for _ in range(maxn+1)]
    for n in range(maxn+1):
        comb_small[n][0] = 1
        for k in range(1, n+1):
            comb_small[n][k] = (comb_small[n-1][k-1] + comb_small[n-1][k]) % MOD

    # precompute factorial and inv factorial up to 7
    maxf = r+1
    fact = [1]*(maxf+1)
    for i in range(1, maxf+1):
        fact[i] = fact[i-1]*i % MOD
    inv_fact = [1]*(maxf+1)
    inv_fact[maxf] = pow(fact[maxf], MOD-2, MOD)
    for i in range(maxf, 0, -1):
        inv_fact[i-1] = inv_fact[i]*i % MOD

    # Precompute partial fractions for t = 1..r+1
    # We want for each t: A_t and B_t[1..t] such that
    # 1/((1-z)*(1 - M z)^t) = A_t/(1 - z) + sum_{j=1..t} B_t[j]/(1 - M z)^j
    A = [0]*(r+2)
    B = [None]*(r+2)  # B[t] is list of length t+1, 1..t used
    invM = pow(M, MOD-2, MOD)
    for t in range(1, r+2):
        # compute A_t
        denom = (1 - M) % MOD
        denom_pow = pow(denom, t, MOD)
        A_t = pow(denom_pow, MOD-2, MOD)
        A[t] = A_t
        # unknowns b[1..t]
        # build matrix eq: for n in 0..t-1
        # sum_{j=1..t} b[j]*mat[n][j] = rhs2[n]
        mat = [[0]*t for _ in range(t)]
        rhs2 = [0]*t
        # precompute powers (-M)^n
        pow_m = [1]*(t+1)
        negM = (-M) % MOD
        for i in range(1, t+1):
            pow_m[i] = pow_m[i-1] * negM % MOD
        for n in range(t):
            # rhs: if n==0 then 1 else 0, minus A_t * C(t,n)*(-M)^n
            rhs = (1 if n == 0 else 0)
            rhs = (rhs - A_t * comb_small[t][n] * pow_m[n]) % MOD
            rhs2[n] = rhs
            for j in range(1, t+1):
                # coeff = C(t-j, n)*(-M)^n - C(t-j, n-1)*(-M)^(n-1)
                c1 = comb_small[t-j][n] if n <= (t-j) else 0
                term1 = c1 * pow_m[n] % MOD
                if n-1 >= 0:
                    c2 = comb_small[t-j][n-1] if (n-1) <= (t-j) else 0
                    term2 = c2 * pow_m[n-1] % MOD
                else:
                    term2 = 0
                mat[n][j-1] = (term1 - term2) % MOD
        # Gaussian elimination to solve mat * b = rhs2
        # mat is t x t
        # augment rhs2
        for i in range(t):
            # find pivot
            piv = i
            while piv < t and mat[piv][i] == 0:
                piv += 1
            if piv == t:
                continue
            if piv != i:
                mat[i], mat[piv] = mat[piv], mat[i]
                rhs2[i], rhs2[piv] = rhs2[piv], rhs2[i]
            inv_p = pow(mat[i][i], MOD-2, MOD)
            for j in range(i, t):
                mat[i][j] = mat[i][j] * inv_p % MOD
            rhs2[i] = rhs2[i] * inv_p % MOD
            for u in range(t):
                if u != i and mat[u][i] != 0:
                    factor = mat[u][i]
                    for j in range(i, t):
                        mat[u][j] = (mat[u][j] - factor * mat[i][j]) % MOD
                    rhs2[u] = (rhs2[u] - factor * rhs2[i]) % MOD
        b = [0]*(t+1)
        for j in range(1, t+1):
            b[j] = rhs2[j-1]
        B[t] = b

    # Precompute M^N and invM_powers
    M_N = pow(M, N, MOD)
    invM = pow(M, MOD-2, MOD)
    invM_pows = [1]*(r+1)
    for i in range(1, r+1):
        invM_pows[i] = invM_pows[i-1] * invM % MOD

    # Now enumerate all partitions of all subsets of {0..r-1}
    res = 0
    P = []  # list of blocks, each block is list of indices

    def dfs(idx):
        nonlocal res
        if idx == r:
            # process partition P
            # compute prod_c, s, t
            prod_c = 1
            s = 0
            for block in P:
                mask = 0
                for i in block:
                    mask |= 1<<i
                val = cB[mask]
                if val == 0:
                    prod_c = 0
                    break
                prod_c = prod_c * val % MOD
                s += len(block)
            if prod_c != 0:
                t = len(P) + 1
                if N >= s:
                    # term1
                    term1 = prod_c * A[t] % MOD
                    res = (res + term1) % MOD
                    # term2
                    # M^{N-s} = M_N * invM^s
                    D = M_N * invM_pows[s] % MOD
                    Bs = B[t]
                    for j in range(1, t+1):
                        rchoose = j-1
                        if rchoose == 0:
                            comb = 1
                        else:
                            # C(N-s + rchoose -1, rchoose -1) with shifting?
                            # Actually we want C(N-s + j -1, j-1) = C((N-s)+rchoose, rchoose)
                            # which is prod_{k=0..rchoose-1} (N-s + k +1)/ (rchoose)!
                            num = 1
                            base = (N - s) % MOD
                            for k in range(rchoose):
                                num = num * ((base + 1 + k) % MOD) % MOD
                            comb = num * inv_fact[rchoose] % MOD
                        add = prod_c * Bs[j] % MOD * comb % MOD * D % MOD
                        res = (res + add) % MOD
            return
        # option: exclude idx
        dfs(idx+1)
        # option: include idx in existing block
        for i in range(len(P)):
            P[i].append(idx)
            dfs(idx+1)
            P[i].pop()
        # option: include as new block
        P.append([idx])
        dfs(idx+1)
        P.pop()

    dfs(0)
    # subtract T_0 = 1
    ans = (res - 1) % MOD
    print(ans)

if __name__ == "__main__":
    main()