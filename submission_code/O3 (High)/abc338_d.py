import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    X = list(map(int, sys.stdin.readline().split()))

    diff = [0] * (N + 2)               # difference array for edges 1 … N
    base = 0                           # length with all bridges

    for a, b in zip(X, X[1:]):
        cw = (b - a) % N               # clockwise edges (1 … N-1)
        if cw == 0:                    # Xk != Xk+1 by constraints
            continue
        ccw = N - cw
        shorter = cw if cw < ccw else ccw
        longer = N - shorter
        delta = longer - shorter       # = |cw - ccw| = N - 2*shorter
        base += shorter

        if delta == 0:                 # both directions equally long
            continue

        if cw < ccw:                   # clockwise path is shorter
            start = a
            length = cw
        else:                          # counter-clockwise path is shorter
            start = b                  # (= Xk+1)
            length = ccw

        l = start
        r = start + length - 1         # inclusive, in 1 … 2N-1

        if r <= N:                     # no wrap
            diff[l] += delta
            diff[r + 1] -= delta
        else:                          # wraps around 1
            diff[l] += delta
            diff[N + 1] -= delta
            end_wrap = r - N           # part 1 … end_wrap
            diff[1] += delta
            diff[end_wrap + 1] -= delta

    best_extra = float('inf')
    cur = 0
    for i in range(1, N + 1):
        cur += diff[i]
        if cur < best_extra:
            best_extra = cur

    print(base + best_extra)


if __name__ == '__main__':
    main()