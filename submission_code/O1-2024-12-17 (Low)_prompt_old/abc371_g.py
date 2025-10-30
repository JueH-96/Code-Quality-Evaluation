def solve():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    P = list(map(lambda x: int(x) - 1, input_data[1:N+1]))  # zero-based
    A = list(map(int, input_data[N+1:2*N+1]))

    # We want to decompose P into cycles.  Each cycle c gives us
    # positions c[0], c[1], ..., c[k-1].
    # Then the operation allows us to "rotate" the values of A
    # in that cycle arbitrarily.
    # We choose the rotation that yields a lexicographically smallest
    # arrangement across the entire array.

    # Because cycles are disjoint in index, we can handle each cycle
    # independently: for the positions in one cycle, we reorder (rotate)
    # their A-values to get the lex. minimal arrangement within those positions.
    # Combining these minimal local choices yields a globally minimal arrangement
    # since the cycles do not overlap in positions.

    visited = [False]*N
    adj = P  # same representation, P[i] = next index

    # Function: Booth's algorithm to find start index of lexicographically
    # minimal rotation in O(k) time, where k = len(arr).
    # Returns the starting index of that minimal rotation.
    def booth_algorithm(arr):
        # Standard Booth's algorithm
        n = len(arr)
        s = arr + arr  # concatenate to itself
        # i = beginning index of current candidate for minimal rotation
        # j = index of the challenger
        # k = current offset while comparing
        i, j, k = 0, 1, 0
        while i < n and j < n and k < n:
            if s[i+k] == s[j+k]:
                k += 1
                continue
            elif s[i+k] > s[j+k]:
                i = i + k + 1
                if i <= j:
                    i = j + 1
            else:
                j = j + k + 1
                if j <= i:
                    j = i + 1
            k = 0
        return min(i, j)

    # We'll store the final arrangement here (1-based indexing in output, but zero-based internally)
    result = [0]*N

    for start in range(N):
        if not visited[start]:
            # Explore the cycle starting at 'start'
            cycle_indices = []
            cur = start
            while not visited[cur]:
                visited[cur] = True
                cycle_indices.append(cur)
                cur = adj[cur]

            # Now 'cycle_indices' is one cycle
            if len(cycle_indices) == 1:
                # length-1 cycle, no rotation needed
                i = cycle_indices[0]
                result[i] = A[i]
                continue

            # Sort the cycle indices so we know the order of positions in the final array
            cycle_indices.sort()
            # Gather the A-values from these positions in ascending order of the positions
            vals = [A[i] for i in cycle_indices]
            # We want the lexicographically minimal rotation of vals.
            shift_start = booth_algorithm(vals)
            # Rotate the array accordingly
            rotated = vals[shift_start:] + vals[:shift_start]
            # Place them in 'result' matching the sorted positions
            for i, pos in enumerate(cycle_indices):
                result[pos] = rotated[i]

    print(" ".join(map(str, result)))