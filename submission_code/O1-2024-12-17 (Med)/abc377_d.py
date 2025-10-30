def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    LR = input_data[2:]
    
    # We want to exclude all (l, r) that completely contain at least one [L_i, R_i].
    # Define S_i to be the set of (l, r) that contain [L_i, R_i], i.e. l <= L_i and r >= R_i.
    # We need |union_i S_i|, then subtract from total pairs = M*(M+1)//2.

    # For a fixed l, (l, r) is in the union if there exists i with L_i >= l and R_i <= r.
    # Let minR(l) = min(R_i for all i such that L_i >= l).
    # Then for a given l, all r >= minR(l) form pairs in the union. That is r in [minR(l), M].
    # Count them: M - minR(l) + 1 if minR(l) <= M, else 0.
    # Summing over l=1..M gives |union_i S_i|, and we do totalPairs - that.

    # Step 1: build an array minRof[x] = min of R_i over all i with L_i = x (or INF if none).
    INF = 10**10
    minRof = [INF] * (M + 1)
    idx = 0
    for _ in range(N):
        L_i = int(LR[idx]); R_i = int(LR[idx + 1])
        idx += 2
        if R_i < minRof[L_i]:
            minRof[L_i] = R_i

    # Step 2: build a suffixMin array so that suffixMin[l] = min_{x >= l} minRof[x].
    # We'll do a right-to-left sweep.
    suffixMin = [INF] * (M + 2)
    for x in range(M, 0, -1):
        suffixMin[x] = min(suffixMin[x + 1], minRof[x])

    # Step 3: sum up the counts of r for each l.
    total_pairs = M * (M + 1) // 2
    union_count = 0
    for l in range(1, M + 1):
        rmin = suffixMin[l]
        if rmin <= M:
            union_count += (M - rmin + 1)

    # Step 4: print the difference
    print(total_pairs - union_count)

# Do not forget to call main()
if __name__ == "__main__":
    main()