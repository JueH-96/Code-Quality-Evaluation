import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    mod = 998244353
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    # Special case M=1
    if M == 1:
        # only sequences of 1's, score always 1, total sequences = N
        print(N % mod)
        return
    # compute primes up to M
    primes = []
    is_prime = [True] * (M+1)
    for i in range(2, M+1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i*i, M+1, i):
                is_prime[j] = False
    P = len(primes)
    # compute exponent vectors for a in 1..M
    # e[a] = list of exponents for each prime
    e = [[0]*P for _ in range(M+1)]
    for a in range(1, M+1):
        tmp = a
        for j, p in enumerate(primes):
            cnt = 0
            while tmp % p == 0:
                tmp //= p
                cnt += 1
            e[a][j] = cnt
    # compute b[mask] = sum_{a=1..M} prod_{j in mask} e[a][j]
    # mask runs from 0 to (1<<P)-1
    maxS = 1 << P
    b = [0] * maxS
    for mask in range(maxS):
        total = 0
        # for each a compute product of e[a][j] for j in mask
        # mask==0 => product empty =1
        if mask == 0:
            total = M
        else:
            # iterate a
            # we can speed by caching if needed, but M small <=16
            for a in range(1, M+1):
                prod = 1
                m = mask
                # iterate bits
                # use while to avoid looping all bits
                # but P<=6 so loop small
                j = 0
                # faster: for j in 0..P-1
                # but we'll do simple
                for j in range(P):
                    if (mask >> j) & 1:
                        prod *= e[a][j]
                        # if prod==0: break early?
                        # But e very small, skip
                total += prod
        b[mask] = total % mod
    # DP[S][d] = sum over ordered partitions of S into d blocks weighted by b
    # DP dims: [maxS][P+1]
    DP = [[0] * (P+1) for _ in range(maxS)]
    DP[0][0] = 1
    # for mask>0
    # we need popcount for mask
    for mask in range(1, maxS):
        pc = mask.bit_count()
        # DP[mask][0] = 0 (already)
        # for d from 1..pc
        # for each d, sum over nonempty submasks T of mask
        # DP[mask][d] = sum_{T subset mask, T!=0} b[T] * DP[mask^T][d-1]
        # we can loop T = mask; T>0; T=(T-1)&mask
        for d in range(1, pc+1):
            s = 0
            sub = mask
            # iterate non-empty submasks
            # because mask small, this loop small
            while sub:
                # sub is nonempty
                s += b[sub] * DP[mask ^ sub][d-1]
                sub = (sub - 1) & mask
            DP[mask][d] = s % mod
    # compute B[d] = sum_{mask: popcount(mask)>=d} DP[mask][d]
    B = [0] * (P+1)
    for d in range(0, P+1):
        s = 0
        for mask in range(maxS):
            if mask.bit_count() >= d:
                s += DP[mask][d]
        B[d] = s % mod
    # Build matrix A of dimension D = P+1
    D = P + 1
    # A is list of lists
    A = [[0] * D for _ in range(D)]
    # f_i(k) = M*f_i(k-1) + f_{i-1}(k-1) for i>=1; f_0(k) = M*f_0(k-1)
    for i in range(D):
        A[i][i] = M % mod
    for i in range(1, D):
        A[i][i-1] = 1
    # initial vector G0 = [f_0(0), f_1(0), ..., f_P(0)] = [1,0,...,0]
    G0 = [0] * D
    G0[0] = 1
    # matrix and vector operations
    def mat_mul(X, Y):
        # multiply X*Y where X,Y are D x D matrices
        Z = [[0] * D for _ in range(D)]
        for i in range(D):
            Xi = X[i]
            Zi = Z[i]
            for k in range(D):
                if Xi[k]:
                    v = Xi[k]
                    Yk = Y[k]
                    # unroll j
                    for j in range(D):
                        Zi[j] = (Zi[j] + v * Yk[j]) % mod
        return Z

    def mat_vec_mul(X, v):
        # X is D x D, v is D vector
        res = [0] * D
        for i in range(D):
            s = 0
            Xi = X[i]
            # manual dot
            for j in range(D):
                s += Xi[j] * v[j]
            res[i] = s % mod
        return res

    def vec_add(u, v):
        # vector addition mod
        return [(u[i] + v[i]) % mod for i in range(D)]

    # recursion to compute (A^n, H_n = sum_{i=1..n}A^i G0, V_n = A^n G0)
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def rec(n):
        if n == 0:
            # P0 = I
            I = [[0]*D for _ in range(D)]
            for i in range(D):
                I[i][i] = 1
            # H0 = zero vector
            H0 = [0]*D
            # V0 = G(0) = G0
            V0 = G0[:]  # copy
            return (I, H0, V0)
        if n & 1:
            # odd
            P_prev, H_prev, V_prev = rec(n-1)
            # P_n = P_prev * A
            Pn = mat_mul(P_prev, A)
            # V_n = A * V_prev
            Vn = mat_vec_mul(A, V_prev)
            # H_n = H_prev + V_n
            Hn = vec_add(H_prev, Vn)
            return (Pn, Hn, Vn)
        else:
            # even
            Ph, Hh, Vh = rec(n >> 1)
            # P_n = Ph * Ph
            Pn = mat_mul(Ph, Ph)
            # H_n = Hh + Ph * Hh
            # first compute Ph * Hh
            PHh = mat_vec_mul(Ph, Hh)
            Hn = vec_add(Hh, PHh)
            # V_n = Ph * Vh
            Vn = mat_vec_mul(Ph, Vh)
            return (Pn, Hn, Vn)

    # compute rec(N) to get H_N
    _, HN, _ = rec(N)
    # answer = sum_{i=0..P} B[i] * HN[i]
    ans = 0
    for i in range(D):
        ans = (ans + B[i] * HN[i]) % mod
    print(ans)

if __name__ == "__main__":
    main()