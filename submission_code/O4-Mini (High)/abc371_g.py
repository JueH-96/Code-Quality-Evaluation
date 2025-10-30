import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N = int(input().strip())
    P = list(map(int, input().split()))
    A = list(map(int, input().split()))
    # convert P to 0-based
    for i in range(N):
        P[i] -= 1

    # build smallest-prime-factor table up to N
    max_n = N
    spf = [0] * (max_n + 1)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, max_n + 1):
        if spf[i] == 0:
            spf[i] = i
            step = i
            for j in range(i * i, max_n + 1, step):
                if spf[j] == 0:
                    spf[j] = i

    # find cycles of P
    visited = [False] * N
    cycles = []
    for i in range(N):
        if not visited[i]:
            cur = []
            j = i
            while not visited[j]:
                visited[j] = True
                cur.append(j)
                j = P[j]
            # cur is a cycle in P-order, but starting at i
            # rotate so that smallest index in cycle is first
            fpc = min(cur)
            k0 = 0
            # find index of fpc
            for idx in range(len(cur)):
                if cur[idx] == fpc:
                    k0 = idx
                    break
            C = cur[k0:] + cur[:k0]
            cycles.append(C)

    # sort cycles by their first element (the minimal index in each cycle)
    cycles.sort(key=lambda c: c[0])

    # e_max[p] = max exponent of prime p in lcm of processed lengths
    # rep_x[p] = t mod p^e_max[p] for that prime's p-power block
    e_max = {}
    rep_x = {}
    # x_offsets for each cycle
    x_offsets = [0] * len(cycles)

    # process cycles in increasing order of first position
    for ci, C in enumerate(cycles):
        L = len(C)
        # factorize L
        Ltemp = L
        factor_map = {}
        while Ltemp > 1:
            p = spf[Ltemp]
            cnt = 0
            while Ltemp % p == 0:
                Ltemp //= p
                cnt += 1
            factor_map[p] = cnt

        # build constraints from gcd of M_prev and this L
        # for each p dividing L with e_m>0, constrain k mod p^e_d = rep_x[p] mod p^e_d
        constraints = []
        for p, e_j in factor_map.items():
            e_m = e_max.get(p, 0)
            if e_m > 0:
                # exponent in gcd = min(e_m, e_j)
                e_d = e_m if e_m < e_j else e_j
                if e_d > 0:
                    mod = p ** e_d
                    r = rep_x[p] % mod
                    constraints.append((mod, r))

        # scan k = 0..L-1 to find minimal A at C[0] (the cycle's minimal index) under constraints
        best_k = 0
        # values in A are 1..N, so N+1 is safe "infinite"
        best_val = N + 1
        if not constraints:
            # no constraint, just find global min in v_j
            for k in range(L):
                val = A[C[k]]
                if val < best_val:
                    best_val = val
                    best_k = k
        else:
            # have constraints; test each k
            for k in range(L):
                ok = True
                for mod, r in constraints:
                    if k % mod != r:
                        ok = False
                        break
                if not ok:
                    continue
                val = A[C[k]]
                if val < best_val:
                    best_val = val
                    best_k = k

        x_offsets[ci] = best_k

        # update e_max and rep_x with prime powers from this cycle's length
        for p, e_j in factor_map.items():
            prev_e = e_max.get(p, 0)
            if e_j > prev_e:
                e_max[p] = e_j
                rep_x[p] = best_k % (p ** e_j)

    # reconstruct the lexicographically smallest A using x_offsets
    res = [0] * N
    for ci, C in enumerate(cycles):
        L = len(C)
        shift = x_offsets[ci]
        # A*(C[t]) = A[C[(t + shift) mod L]]
        for t in range(L):
            pos = C[t]
            src = C[(t + shift) % L]
            res[pos] = A[src]

    # output
    out = sys.stdout
    out.write(" ".join(str(v) for v in res))
    out.write("
")

if __name__ == "__main__":
    main()