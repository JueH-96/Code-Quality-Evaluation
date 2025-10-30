def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    P = list(map(lambda x: int(x) - 1, input_data[1:1+N]))  # zero-based
    A = list(map(lambda x: int(x) - 1, input_data[1+N:1+2*N]))  # zero-based

    visited = [False]*N
    newA = [None]*N

    # Standard linear-time "minimum rotation" comparison
    # but adapted to our custom indexing:
    # X_s[k] = cycVal[ (B[k] + s) mod L ].
    def find_min_shift(cycVal, B):
        L = len(cycVal)
        i, j, k = 0, 1, 0
        while i < L and j < L and k < L:
            xi = cycVal[(B[k] + i) % L]
            xj = cycVal[(B[k] + j) % L]
            if xi == xj:
                k += 1
            else:
                if xi < xj:
                    j = j + k + 1
                else:
                    i = i + k + 1
                if i == j:
                    j += 1
                k = 0
        return min(i, j)

    # Find cycles in P
    for start in range(N):
        if not visited[start]:
            cycle_pos = []
            cur = start
            while not visited[cur]:
                visited[cur] = True
                cycle_pos.append(cur)
                cur = P[cur]

            L = len(cycle_pos)
            if L == 1:
                # Single-element cycle - no rotation possible
                newA[cycle_pos[0]] = A[cycle_pos[0]]
                continue

            # Gather values and positions for this cycle
            cycVal = [A[idx] for idx in cycle_pos]
            sorted_cpos = sorted(cycle_pos)
            idxMap = {}
            for j, cidx in enumerate(cycle_pos):
                idxMap[cidx] = j

            # B[k] = index-in-cycle of the k-th smallest position
            B = [idxMap[pos] for pos in sorted_cpos]

            # Find best shift for lex-min
            sMin = find_min_shift(cycVal, B)

            # Apply that shift in the cycle order
            for m in range(L):
                newA[cycle_pos[m]] = cycVal[(m + sMin) % L]

    # Output (convert back to 1-based)
    print(' '.join(str(x+1) for x in newA))

# Call main() as required
if __name__ == "__main__":
    main()