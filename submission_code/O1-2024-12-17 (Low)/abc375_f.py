def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast I/O cursor
    idx = 0

    # Read initial inputs
    N = int(input_data[idx]); idx += 1
    M = int(input_data[idx]); idx += 1
    Q = int(input_data[idx]); idx += 1

    edges = []
    for _ in range(M):
        A = int(input_data[idx]); idx += 1
        B = int(input_data[idx]); idx += 1
        C = int(input_data[idx]); idx += 1
        # Store edges as (A, B, C), zero-based
        edges.append((A - 1, B - 1, C))

    # We will process queries in two passes:
    # 1) First pass: we record how many road-closures happened before each query of type 2,
    #    and also record at which "closure index" each road gets closed.
    # 2) Second pass: we build an all-pairs distance matrix once per "state" (i.e., after k closures),
    #    and answer all queries of type 2 that belong to that state.

    # block_of_edge[i] = (integer) the earliest closure-block at which edge i is closed.
    # if an edge is never closed, we set block_of_edge[i] = T+1 (where T is total closures).
    block_of_edge = [None]*M
    # Collect the order in which roads become closed
    close_order = []  # will hold (road_index) in the order they are closed

    queries = []      # will hold (type, x, y) or (type, i) in 1-based input form
    # queries_by_block will store a list of queries of type 2 for each block
    # block k means: after k closures have occurred (0 <= k <= T)
    # We don't know T until we read all type-1 queries.
    # We'll store them temporarily and then distribute them into blocks once we know
    # how many closures up to each point in time.

    for _ in range(Q):
        t = int(input_data[idx]); idx += 1
        if t == 1:
            i = int(input_data[idx]); idx += 1
            queries.append((1, i))
        else:
            x = int(input_data[idx]); idx += 1
            y = int(input_data[idx]); idx += 1
            queries.append((2, x, y))

    # Initialize block_of_edge for each edge to a large sentinel: we use T+1 eventually
    # but T is unknown, so just keep None for now; we'll fill it in after reading closures.
    for i in range(M):
        block_of_edge[i] = None

    # Determine block_of_edge by the order edges are closed
    # Also we will keep track of how many closures have occurred so far
    # so we can eventually assign queries of type 2 to the correct block.
    closure_count = 0
    for q in queries:
        if q[0] == 1:
            # road i becomes closed
            road_i = q[1] - 1  # zero-based
            block_of_edge[road_i] = closure_count
            closure_count += 1
    # T is the total number of closures
    T = closure_count

    # For any edge *never* closed, set block_of_edge[i] = T (meaning it remains open up to block T-1,
    # and is effectively "closed" at block T).
    for i in range(M):
        if block_of_edge[i] is None:
            block_of_edge[i] = T

    # Now we know T, we can distribute the type-2 queries into blocks based on how many closures
    # have occurred so far while reading them in order.
    queries_by_block = [[] for _ in range(T+1)]
    answers = [None]*Q  # to store answers in the original order
    current_closures = 0  # how many closures have happened so far
    q_index = 0           # index over all queries

    for q in queries:
        if q[0] == 1:
            # closure query
            # after we see it, the closure_count increments
            # so it affects future queries
            road_i = q[1] - 1
            current_closures += 1
            q_index += 1
        else:
            # query type 2
            x, y = q[1], q[2]
            # that belongs to block "current_closures"
            # record it
            queries_by_block[current_closures].append((q_index, x - 1, y - 1))
            q_index += 1

    # We will build a distance matrix for each block k = 0..T
    # block k means: all edges i where block_of_edge[i] >= k remain open.
    # Then we run Floyd-Warshall (or any all-pairs method) for that graph
    # and answer queries in queries_by_block[k].

    import math

    INF = 10**18

    # Prepare a helper to build adjacency and run Floyd-Warshall
    # We do this T+1 times. Each time, we consider edges i with block_of_edge[i] >= k.

    # Instead of re-building from scratch each time, we will do so from scratch anyway
    # because partial removal is tricky to do incrementally. N <= 300 is small enough
    # that this might succeed if optimized carefully.

    # Build once per block
    # cost: (T+1)*O(N^3) which is up to 301*27,000,000 ~ 8.1e9 operations (very large in Python).
    # We'll try to implement quickly and hope it passes with efficient usage.

    # To reduce overhead, we wrap the FW in a function with local variable lookups.

    def floyd_warshall_build_and_run(block_k):
        # Build dist matrix
        dist = [[INF]*N for _ in range(N)]
        for i in range(N):
            dist[i][i] = 0
        # Add edges that are still open at block_k
        for (A, B, C) in edges:
            if block_of_edge[edges.index((A,B,C))] >= block_k:
                # Edge is open
                if C < dist[A][B]:
                    dist[A][B] = C
                    dist[B][A] = C
        # Floyd Warshall
        for k in range(N):
            dk = dist[k]
            for i in range(N):
                di = dist[i]
                alt_ik = di[k]
                if alt_ik == INF:
                    continue
                for j in range(N):
                    current = di[j]
                    candidate = alt_ik + dk[j]
                    if candidate < current:
                        di[j] = candidate
        return dist

    # The above approach with edges.index(...) is O(M) each time, done M times => O(M^2)=2.0e9 worst, too big
    # We must not do edges.index(...) in a loop. Instead, we should just iterate over the numeric index i of edges.

    def floyd_warshall_with_block(block_k):
        dist = [[INF]*N for _ in range(N)]
        for i in range(N):
            dist[i][i] = 0
        for i in range(M):
            if block_of_edge[i] >= block_k:
                A, B, C = edges[i]
                # Insert edge
                if C < dist[A][B]:
                    dist[A][B] = C
                    dist[B][A] = C
        for k in range(N):
            dk = dist[k]
            for i in range(N):
                di = dist[i]
                alt_ik = di[k]
                if alt_ik == INF:
                    continue
                sum_k = dk  # local ref
                for j in range(N):
                    candidate = alt_ik + sum_k[j]
                    if candidate < di[j]:
                        di[j] = candidate
        return dist

    # Now compute for each block once, then answer queries for that block
    # We'll store each dist in a list to avoid re-computing, since queries might interleave.
    # But that would require a lot of memory if we store all T+1 NxN matrices. T up to 300,
    # NxN = 300*300 = 90k, so 300 * 90k = 27,000,000 integers, which might be large but in principle feasible.
    #
    # Alternatively, we can build the matrix, answer the queries, then discard it. That saves memory,
    # but we must do all queries for that block immediately (which is what we planned: queries_by_block).
    # Then we move on. That is likely better for memory.

    # Let's implement that now:

    for k in range(T+1):
        distk = floyd_warshall_with_block(k)
        # answer queries in queries_by_block[k]
        for (original_idx, x, y) in queries_by_block[k]:
            ans = distk[x][y]
            if ans >= INF:
                answers[original_idx] = -1
            else:
                answers[original_idx] = ans

    # Finally, print the answers for all queries of type 2 in the correct order (they are exactly Q lines
    # but type-1 queries produce no output, so only type-2 queries do).
    # We only need to print answers for queries whose type=2. The others remain None.
    import sys
    out = []
    q_idx = 0
    for q in queries:
        if q[0] == 2:
            out.append(str(answers[q_idx]))
        q_idx += 1

    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()