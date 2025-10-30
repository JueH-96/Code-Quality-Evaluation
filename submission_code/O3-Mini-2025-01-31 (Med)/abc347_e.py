def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    Q = int(next(it))
    queries = [int(next(it)) for _ in range(Q)]
    
    # L will store, for each query i (1-indexed), the current size of S after the toggle, i.e.
    # the value added later to each element of S.
    L = [0] * (Q + 1)
    
    # in_set[j] indicates whether index j is currently in S (j goes from 1 to N)
    in_set = [False] * (N + 1)
    # last_insertion[j] will record the query index at which j was inserted (toggled on)
    # and is still active (has not been removed yet).
    last_insertion = [-1] * (N + 1)
    # intervals[j] will collect intervals (l, r) (both query indices, inclusive)
    # during which j was present in S and therefore received the addition.
    intervals = [[] for _ in range(N + 1)]
    
    current_size = 0
    # Process queries (we label queries from 1 to Q)
    for i, x in enumerate(queries, start=1):
        if not in_set[x]:
            # x is not in S, so insert it.
            in_set[x] = True
            current_size += 1
            # Record the query index when it was inserted.
            last_insertion[x] = i
        else:
            # x is in S, so remove it.
            # Its active interval started at last_insertion[x] and ends at (i-1)
            start = last_insertion[x]
            # Only add the interval if it is valid (should hold because removal happens after at least one query)
            if start <= i - 1:
                intervals[x].append((start, i - 1))
            in_set[x] = False
            current_size -= 1
            last_insertion[x] = -1
        # After the toggle, S has size current_size and each element currently in S gets addition = current_size.
        L[i] = current_size

    # For any index j still in S after the final query,
    # the active interval goes until Q (the last query).
    for j in range(1, N + 1):
        if in_set[j]:
            intervals[j].append((last_insertion[j], Q))
    
    # Build prefix sum P for L: P[i] = L[1] + L[2] + ... + L[i] (with P[0] = 0).
    P = [0] * (Q + 1)
    for i in range(1, Q + 1):
        P[i] = P[i - 1] + L[i]
    
    # Now, for each index j, A_j is the sum of L[i] over all queries i in the intervals when j was in S.
    # That is: for each interval (l, r) in intervals[j], A[j] gets added: P[r] - P[l-1].
    A = [0] * (N + 1)
    for j in range(1, N + 1):
        total = 0
        for (l, r) in intervals[j]:
            total += P[r] - P[l - 1]
        A[j] = total
    
    # Output the resulting sequence A_1, A_2, …, A_N (ignoring index 0).
    sys.stdout.write(" ".join(str(A[i]) for i in range(1, N + 1)))

if __name__ == '__main__':
    main()