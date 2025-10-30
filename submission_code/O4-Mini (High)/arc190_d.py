import sys
import threading
def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    p = int(next(it))
    A = [[0]*N for _ in range(N)]
    K = 0
    for i in range(N):
        for j in range(N):
            v = int(next(it))
            A[i][j] = v
            if v == 0:
                K += 1
    # base factor = (-1)^K mod p
    if K & 1:
        base = p-1
    else:
        base = 1
    # build M_fixed
    M_fixed = [[0]*N for _ in range(N)]
    if p == 2:
        # build M for p==2: zeros->1 else A
        M = [[0]*N for _ in range(N)]
        for i in range(N):
            Ai = A[i]
            Mi = M[i]
            for j in range(N):
                if Ai[j] != 0:
                    Mi[j] = Ai[j] & 1
                else:
                    Mi[j] = 1
        # multiply M*M mod 2
        mod = 2
        # mat_mul short for p==2
        C = [[0]*N for _ in range(N)]
        # we do C = M^2
        # compute C = M * M
        for i in range(N):
            Ci = C[i]
            Mi = M[i]
            for k in range(N):
                mul = Mi[k]
                if mul:
                    Bk = M[k]
                    # add Bk to Ci
                    for j in range(N):
                        Ci[j] ^= Bk[j]  # since mod 2, addition is xor
        # C now holds T
        # apply base factor
        if base != 1:
            # base == -1 mod2 == 1 anyway, so no change
            pass
        out = C
    elif p == 3:
        # build M_fixed
        for i in range(N):
            Ai = A[i]
            Mi = M_fixed[i]
            for j in range(N):
                v = Ai[j]
                if v != 0:
                    Mi[j] = v % 3
        # define mat_mul mod 3 with postponed row-mod
        def mat_mul_3(X, Y):
            Nloc = N
            # build zero C
            C = [[0]*Nloc for _ in range(Nloc)]
            for i in range(Nloc):
                Ci = C[i]
                Xi = X[i]
                for k in range(Nloc):
                    mul = Xi[k]
                    if mul:
                        Yk = Y[k]
                        # inner add
                        for j in range(Nloc):
                            Ci[j] += mul * Yk[j]
                # row mod
                for j in range(Nloc):
                    Ci[j] %= 3
            return C
        # compute M_fixed^3
        # M2 = M_fixed^2
        M2 = mat_mul_3(M_fixed, M_fixed)
        M3 = mat_mul_3(M2, M_fixed)
        # T = M3
        T = M3
        # add contributions C into T in-place
        # for each zero-edge (u,v)
        for u in range(N):
            for v in range(N):
                if A[u][v] == 0:
                    if u == v:
                        # self-loop
                        # f at pos1: x->u
                        for x in range(N):
                            a = A[x][v]
                            if a:
                                T[x][v] = (T[x][v] + (a % 3)) % 3
                        # f at pos3: u->y
                        for y in range(N):
                            a = A[u][y]
                            if a:
                                T[u][y] = (T[u][y] + (a % 3)) % 3
                    else:
                        # non-loop z; need f = (v->u) fixed-edge
                        a = A[v][u]
                        if a:
                            T[u][v] = (T[u][v] + (a % 3)) % 3
        # apply base factor
        if base != 1:
            # base == -1 mod3 == 2
            for i in range(N):
                Ti = T[i]
                for j in range(N):
                    Ti[j] = (3 - Ti[j]) % 3
        out = T
    else:
        # p >= 4
        # build M_fixed mod p
        for i in range(N):
            Ai = A[i]
            Mi = M_fixed[i]
            for j in range(N):
                v = Ai[j]
                if v != 0:
                    Mi[j] = v % p
        # matrix multiplication mod p with postponed row-mod
        def mat_mul(X, Y):
            Nloc = N
            mod = p
            C = [[0]*Nloc for _ in range(Nloc)]
            for i in range(Nloc):
                Ci = C[i]
                Xi = X[i]
                for k in range(Nloc):
                    mul = Xi[k]
                    if mul:
                        Yk = Y[k]
                        for j in range(Nloc):
                            Ci[j] += mul * Yk[j]
                # row mod
                for j in range(Nloc):
                    Ci[j] %= mod
            return C
        # fast exponentiation M_fixed^p mod p
        def mat_pow(mat, e):
            # result = I
            Nloc = N
            res = [[0]*Nloc for _ in range(Nloc)]
            for i in range(Nloc):
                res[i][i] = 1
            base_mat = mat
            while e:
                if e & 1:
                    res = mat_mul(res, base_mat)
                base_mat = mat_mul(base_mat, base_mat)
                e >>= 1
            return res
        T = mat_pow(M_fixed, p)
        # add contributions for zero-self-loops
        for u in range(N):
            if A[u][u] == 0:
                # self-loop at u
                # f at pos1: x->u
                for x in range(N):
                    a = A[x][u]
                    if a:
                        T[x][u] = (T[x][u] + a) % p
                # f at pos p: u->y
                for y in range(N):
                    a = A[u][y]
                    if a:
                        T[u][y] = (T[u][y] + a) % p
        # apply base factor
        if base != 1:
            for i in range(N):
                Ti = T[i]
                for j in range(N):
                    Ti[j] = (p - Ti[j]) % p
        out = T
    # print result
    w = sys.stdout.write
    for i in range(N):
        row = out[i]
        # ensure each entry mod p
        if p > 3:
            # already mod p
            w(" ".join(str(x) for x in row))
        else:
            # ensure mod p
            w(" ".join(str(x % p) for x in row))
        w("
")

if __name__ == "__main__":
    main()