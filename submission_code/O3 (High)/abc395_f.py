import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    X = int(next(it))

    U = []
    D = []
    s_sum = 0
    s_min = 10 ** 20      # large enough

    for _ in range(N):
        u = int(next(it)); d = int(next(it))
        U.append(u)
        D.append(d)
        s = u + d
        s_sum += s
        if s < s_min:
            s_min = s                          # minimal possible H

    # ----- feasibility test for a fixed H ---------------------------------
    def feasible(H: int) -> bool:
        lo = max(0, H - D[0])                  # interval for U_1'
        hi = min(U[0], H)
        if lo > hi:
            return False
        for i in range(1, N):
            Li = max(0, H - D[i])
            Ri = min(U[i], H)
            lo = max(Li, lo - X)               # propagate constraints
            hi = min(Ri, hi + X)
            if lo > hi:
                return False
        return True
    # ----------------------------------------------------------------------

    # binary search for the maximum feasible H
    low, high = 0, s_min
    while low < high:
        mid = (low + high + 1) // 2
        if feasible(mid):
            low = mid
        else:
            high = mid - 1

    H_best = low
    answer = s_sum - H_best * N
    print(answer)

if __name__ == "__main__":
    main()