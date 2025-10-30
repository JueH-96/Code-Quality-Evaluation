import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1000000)
    input = sys.stdin.readline
    MOD = 998244353

    N, M = map(int, input().split())
    B = list(map(int, input().split()))
    # 1-based for convenience
    # Precompute prefix minimal fixed and prefix unknown counts
    c1 = [M+1] * (N+2)  # c1[k] = min fixed in B[1..k-1], default M+1 if none
    up1 = [0] * (N+2)   # up1[k] = count of -1 in B[1..k-1]
    for k in range(2, N+1):
        prev_min = c1[k-1]
        prev_unknown = up1[k-1]
        bi = B[k-2]
        if bi == -1:
            c1[k] = prev_min
            up1[k] = prev_unknown + 1
        else:
            # fixed value
            if bi < prev_min:
                c1[k] = bi
            else:
                c1[k] = prev_min
            up1[k] = prev_unknown

    # Precompute suffix maximal fixed and suffix unknown counts
    d2 = [0] * (N+2)   # d2[k] = max fixed in B[k..N], default 0 if none
    up2 = [0] * (N+2)  # up2[k] = count of -1 in B[k..N]
    for k in range(N, 0, -1):
        nxt_max = d2[k+1]
        nxt_unknown = up2[k+1]
        bi = B[k-1]
        if bi == -1:
            d2[k] = nxt_max
            up2[k] = nxt_unknown + 1
        else:
            # fixed value
            if bi > nxt_max:
                d2[k] = bi
            else:
                d2[k] = nxt_max
            up2[k] = nxt_unknown

    # total unknowns
    total_unknown = up1[N+1]  # up1[N+1] is # -1 in B[1..N]
    # sum over k of assignments where prefix[1..k-1] and suffix[k..N] are disconnected
    sumE = 0
    for k in range(2, N+1):
        # prefix = 1..k-1, suffix = k..N
        u1 = up1[k]
        u2 = up2[k]
        # need all suffix values <= t, prefix values >= t+1
        # fixed suffix max d2[k] <= t  => t >= d2[k]
        # fixed prefix min c1[k] >= t+1 => t <= c1[k]-1
        L = d2[k]
        if L < 1:
            L = 1
        R = c1[k] - 1
        if R > M-1:
            R = M-1
        if L > R:
            continue
        # sum over t in [L..R] of t^u2 * (M-t)^u1
        # compute with pow mod
        for t in range(L, R+1):
            # compute t^u2 % and (M-t)^u1 %
            # pow(x,0,MOD) is 1, covers that case
            val1 = pow(t, u2, MOD)
            val2 = pow(M - t, u1, MOD)
            sumE += val1 * val2
            if sumE >= 1 << 63:
                sumE %= MOD
    sumE %= MOD

    # plus total assignments M^total_unknown
    total = pow(M, total_unknown, MOD)
    ans = (total + sumE) % MOD
    print(ans)

if __name__ == "__main__":
    main()