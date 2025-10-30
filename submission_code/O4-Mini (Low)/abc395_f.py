import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N, X = map(int, input().split())
    U = [0]*N
    D = [0]*N
    total = 0
    min_sum = 10**30
    for i in range(N):
        u, d = map(int, input().split())
        U[i] = u
        D[i] = d
        total += u + d
        s = u + d
        if s < min_sum:
            min_sum = s

    # Check if a given H is feasible.
    def feasible(H):
        # We need to pick UI' in [Li, Ri] with adjacent diff <= X
        # where Li = max(H - D[i], 0), Ri = U[i].
        # We do a forward sweep maintaining current feasible [curL, curR].
        # At i=0:
        L0 = H - D[0]
        if L0 < 0:
            L0 = 0
        curL = L0
        curR = U[0]
        if curL > curR:
            return False
        # For each subsequent tooth:
        for i in range(1, N):
            Li = H - D[i]
            if Li < 0:
                Li = 0
            Ri = U[i]
            # Because |U'i - U'i-1| <= X, we intersect [Li, Ri] with [curL-X, curR+X].
            low = curL - X
            high = curR + X
            if low > Ri or high < Li:
                return False
            # Update current range
            curL = max(Li, low)
            curR = min(Ri, high)
        return True

    # Binary search maximum H in [0, min_sum] such that feasible(H) is True.
    lo = 0
    hi = min_sum
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if feasible(mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Minimum cost = total reductions = sum(Ui+Di) - N * best_H
    answer = total - best * N
    print(answer)

if __name__ == "__main__":
    main()