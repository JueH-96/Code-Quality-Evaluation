import sys
import threading

def main():
    import sys

    data = sys.stdin
    line = data.readline().split()
    N = int(line[0])
    X = int(line[1])
    U = [0]*N
    D = [0]*N
    S_min = 10**30
    total = 0
    for i in range(N):
        u,d = map(int, data.readline().split())
        U[i] = u
        D[i] = d
        s = u + d
        total += s
        if s < S_min:
            S_min = s

    # Check if for a given H, we can pick u[i] in [L_i, R_i] with |u[i]-u[i-1]| <= X
    def feasible(H):
        # initial interval at i=0
        # L0 = max(0, H - D[0]); R0 = min(U[0], H)
        low = H - D[0]
        if low < 0:
            low = 0
        high = U[0]
        if high > H:
            high = H
        if low > high:
            return False
        for i in range(1, N):
            # compute L_i and R_i
            Li = H - D[i]
            if Li < 0:
                Li = 0
            Ri = U[i]
            if Ri > H:
                Ri = H
            # expand previous interval by +/- X
            nl = low - X
            nh = high + X
            if nl < Li:
                nl = Li
            if nh > Ri:
                nh = Ri
            if nl > nh:
                return False
            low, high = nl, nh
        return True

    # binary search for max H in [0, S_min]
    lo = 0
    hi = S_min + 1
    # we know H=0 is always feasible
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            lo = mid
        else:
            hi = mid

    H_best = lo
    # cost = sum(U_i+D_i) - N * H
    ans = total - N * H_best
    print(ans)

if __name__ == "__main__":
    main()