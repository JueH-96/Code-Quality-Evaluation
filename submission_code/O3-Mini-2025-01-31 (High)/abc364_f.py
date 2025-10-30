def main():
    import sys,heapq
    # read input as bytes and split (this is fast enough for 2*10^5 numbers)
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    try:
        N = int(next(it))
        Q = int(next(it))
    except StopIteration:
        return

    # Each query is specified by (L, R, C).
    # For query i the op–vertex is (N+i),
    # and it gives an edge (with cost C) from vertex (N+i) to every base vertex j with L<=j<=R.
    queries = [None] * Q
    sum_op = 0  # We'll need to add one copy of cost C for each op vertex.
    for qi in range(Q):
        L = int(next(it))
        R = int(next(it))
        C = int(next(it))
        queries[qi] = (L, R, C)
        sum_op += C

    # Sort queries by their L value.
    queries.sort(key=lambda t: t[0])
    
    # We now want, for each gap j = 1,2,...,N-1,
    # the minimum cost among all queries that cover gap j.
    # A query (L,R,C) covers gap j if L <= j and j < R.
    gap_cost_sum = 0
    active = []  # min-heap of (C, R) for queries that are "active"
    qi = 0  # pointer into sorted queries
    for j in range(1, N):
        # Add all queries that “start” at or before j.
        while qi < Q and queries[qi][0] <= j:
            L, R, C = queries[qi]
            heapq.heappush(active, (C, R))
            qi += 1
        # Remove any queries which no longer cover gap j (i.e. queries with R <= j)
        while active and active[0][1] <= j:
            heapq.heappop(active)
        if not active:
            # No query covers gap j; the base vertices aren’t fully connected.
            sys.stdout.write("-1")
            return
        # The top of the heap corresponds to the minimum cost among queries covering gap j.
        gap_cost_sum += active[0][0]
    
    # If the graph is connected, the MST cost is the sum of:
    #   – the cost to connect each op–vertex (one edge per query): sum_op
    #   – plus the cost to connect the base vertices (one edge per gap): gap_cost_sum
    ans = sum_op + gap_cost_sum
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()  # Don't forget to call main()!