def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+2*N]))

    # ----------------------------------------------------------------
    # 1) Special handling for M = 2.
    #
    #    When M=2, the only "good" sequences of length N≥2 alternate
    #    (e.g. 0,1,0,1,... or 1,0,1,0,...). Any single-step change
    #    would immediately create two adjacent equal values, so no
    #    moves are possible. The only way the answer can be 0 is if
    #    A == B; otherwise it's -1.
    # ----------------------------------------------------------------
    if M == 2:
        # If they are already equal, 0 operations. Otherwise impossible.
        if A == B:
            print(0)
        else:
            print(-1)
        return

    # ----------------------------------------------------------------
    # 2) For M >= 3, it is always possible to transform A into B
    #    (since both are "good") by carefully scheduling increments/
    #    decrements. However, sometimes an extra "detour" step is
    #    required to avoid a temporary conflict with a neighbor.
    #
    #    - The base cost is the sum of the minimal modular distances
    #      dist(A[i], B[i]) over all i.
    #
    #    - Additional "overhead" arises precisely when two neighbors
    #      (i, i+1) are effectively swapping their colors:
    #         A_i => B_i = A_{i+1}
    #         A_{i+1} => B_{i+1} = A_i
    #      In that case, trying to do each one's direct 1-step move
    #      first would cause an adjacency collision. One of them must
    #      take at least one intermediate step ("detour"), which adds
    #      +1 to the total cost.
    #
    #    - If multiple "swap" edges share a node, a single detour
    #      step at that node can fix both edges at once. Hence the
    #      number of extra steps is the size of the minimum vertex
    #      cover of these "swapped" edges, which in a simple chain
    #      is the same as the size of the maximum matching in that
    #      set of edges.
    #
    #  Algorithm:
    #    sum_dist = sum of minimal distances dist(A[i], B[i])
    #    Build a boolean array swapped[] for i=1..N-1:
    #       swapped[i] = True if (B_i, B_{i+1}) == (A_{i+1}, A_i)
    #    overhead = size of maximum matching of consecutive True edges
    #    answer = sum_dist + overhead
    # ----------------------------------------------------------------

    # Function for minimal modular distance
    def dist_mod(x, y):
        d = abs(x - y)
        return min(d, M - d)

    # Sum of all minimal distances
    sum_dist = 0
    for i in range(N):
        sum_dist += dist_mod(A[i], B[i])

    # Identify "swapped" edges
    swapped = [False]*(N-1)
    for i in range(N-1):
        if B[i] == A[i+1] and B[i+1] == A[i]:
            swapped[i] = True

    # Compute overhead = size of maximum matching among these swapped edges
    overhead = 0
    i = 0
    while i < N-1:
        if swapped[i]:
            overhead += 1
            i += 2  # skip the next edge because we matched this one
        else:
            i += 1

    # Final answer
    print(sum_dist + overhead)

# IMPORTANT: Do not forget to call main()
if __name__ == "__main__":
    main()