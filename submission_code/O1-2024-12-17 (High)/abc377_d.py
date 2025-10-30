def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    INF = M + 1

    # minRAt[L] will hold the minimum R among all intervals that start at L
    minRAt = [INF] * (M + 2)

    idx = 2
    for _ in range(N):
        L, R = map(int, data[idx:idx+2])
        idx += 2
        if R < minRAt[L]:
            minRAt[L] = R

    # F[l] = minimum over minRAt[x] for x >= l
    # We'll compute this via a suffix-min array
    F = [INF] * (M + 2)
    for l in range(M, 0, -1):
        F[l] = min(F[l+1], minRAt[l])

    # Total number of (l,r) pairs with 1 <= l <= r <= M
    total_pairs = M * (M + 1) // 2

    # union_count = number of (l,r) pairs that contain at least one interval
    union_count = 0
    for l in range(1, M+1):
        # If F[l] is INF, it means there is no interval [L_i, R_i] with L_i >= l
        # => no interval can be fully contained by intervals starting at/after l
        if F[l] == INF:
            continue
        r_start = max(l, F[l])
        if r_start <= M:
            union_count += (M - r_start + 1)

    # Answer: total pairs minus those that contain at least one interval
    print(total_pairs - union_count)

# Do not forget to call main()!
if __name__ == "__main__":
    main()