def main():
    import sys
    input = sys.stdin.readline

    N, M, X1 = map(int, input().split())
    A = [0] * (M + 1)
    B = [0] * (M + 1)
    S = [0] * (M + 1)
    T = [0] * (M + 1)
    for i in range(1, M + 1):
        a, b, s, t = map(int, input().split())
        A[i] = a
        B[i] = b
        S[i] = s
        T[i] = t

    # Build events: (time, type, train_index)
    # type = 0 for arrival, 1 for departure, so arrivals at the same time
    # are processed before departures.
    events = []
    for i in range(1, M + 1):
        events.append((T[i], 0, i))
        events.append((S[i], 1, i))
    events.sort()

    # M_c[c] = maximum over arrivals at city c of (dist[i] + T[i])
    neg_inf = -10**30
    M_c = [neg_inf] * (N + 1)

    # dist[i] = X_i, the delay for train i.
    dist = [0] * (M + 1)
    dist[1] = X1  # X_1 is given

    # Sweep through events in time order
    for time, typ, i in events:
        if typ == 0:
            # arrival of train i at city B[i]
            c = B[i]
            val = dist[i] + time  # dist[i] + T[i]
            if val > M_c[c]:
                M_c[c] = val
        else:
            # departure of train i from city A[i]
            c = A[i]
            mc = M_c[c]
            if mc != neg_inf:
                # need dist[i] >= mc - S[i]
                cand = mc - time
                if cand > dist[i]:
                    dist[i] = cand

    # Output X_2 ... X_M
    print(" ".join(map(str, dist[2:])))

if __name__ == "__main__":
    main()