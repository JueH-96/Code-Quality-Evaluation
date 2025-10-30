import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    INF_NEG = -10 ** 30          # sufficiently small number

    # read input
    it = iter(sys.stdin.buffer.read().strip().split())
    N = int(next(it))
    M = int(next(it))
    X1 = int(next(it))

    A = [0] * (M + 1)
    B = [0] * (M + 1)
    S = [0] * (M + 1)
    T = [0] * (M + 1)

    events = []                  # (time, kind, train_id)  kind: 0 = arrival, 1 = departure
    for i in range(1, M + 1):
        a = int(next(it)); b = int(next(it))
        s = int(next(it)); t = int(next(it))
        A[i], B[i], S[i], T[i] = a, b, s, t
        events.append((s, 1, i))   # departure  (processed after arrivals at same time)
        events.append((t, 0, i))   # arrival    (processed first when times are equal)

    # sort by time, arrivals before departures when equal
    events.sort()

    # P[c] : current maximum value of (X_i + T_i) among trains that have arrived at city c
    P = [INF_NEG] * (N + 1)

    X = [0] * (M + 1)             # answer array (1-indexed)

    for time, kind, idx in events:
        if kind == 1:             # departure of train idx
            if idx == 1:
                X[idx] = X1
            else:
                need = P[A[idx]] - S[idx]
                if need < 0:
                    need = 0
                X[idx] = need
        else:                     # arrival of train idx
            val = X[idx] + T[idx]
            if val > P[B[idx]]:
                P[B[idx]] = val

    # output X_2 … X_M
    print(' '.join(str(X[i]) for i in range(2, M + 1)))


if __name__ == '__main__':
    main()