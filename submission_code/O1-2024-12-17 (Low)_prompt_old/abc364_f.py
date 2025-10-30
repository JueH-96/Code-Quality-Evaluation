def solve():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    intervals = []
    idx = 2
    for i in range(1, Q+1):
        L = int(input_data[idx]); R = int(input_data[idx+1]); C = int(input_data[idx+2])
        idx += 3
        intervals.append((L, R, C, i))
    
    # We want to determine whether all of the vertices 1..N+Q are in one connected component
    # after adding edges (N+i) -- (j) with cost C_i for j in [L_i, R_i]. If it's connected,
    # we need the cost of the MST.
    #
    # Approach:
    # 1) For each j in [1..N], pick the interval i_j among those covering j that has the minimum cost C_i.
    #    If there is no interval covering j, the graph is disconnected => print -1 and return.
    #    Otherwise, we'll plan to use the edge (j, N + i_j) with cost = min_cost.
    #    Summation of these edges is part of the MST, because for a given j the cheapest edge
    #    to connect j must be in any MST.
    # 2) After adding those N edges, some of the Q "right-side" vertices (N+1..N+Q) might already
    #    be connected into the MST. For each right-side vertex i (i.e., node N+i), if it is not
    #    yet in the MST (i.e., not in the same DS component as the rest), we connect it using
    #    one edge to any j in [L_i..R_i]. The cost of that edge is C_i (the same for all j in that range),
    #    so we can just pick j = L_i for convenience (provided that j was covered in the first place and
    #    is thus already in the MST). If that merges two different DS components, we add C_i to the MST cost.
    # 3) Finally, check if the entire graph is one connected component. If not, print -1; otherwise print
    #    the sum of the chosen edges' costs.
    #
    # The main challenge is to efficiently find the minimum cost covering each j in [1..N].
    # We'll use a sweep-line with a priority queue over intervals:
    # - Sort intervals by L in ascending order.
    # - Iterate j from 1 to N:
    #     * Add all intervals whose L == j to a min-heap keyed by cost, but store also the R boundary.
    #     * Pop from the heap all intervals whose R < j (they no longer cover j).
    #     * The top of the heap, if any, is the interval with the smallest cost covering j.
    #       That cost becomes min_c[j], and we store the index of that interval as best_i[j].
    # - If at some j the heap is empty, there's no covering interval => disconnected => -1.
    #
    # This step is O(N log Q), which should be feasible for N, Q <= 2e5 with efficient Python code.

    import heapq

    # Sort intervals by L
    intervals.sort(key=lambda x: x[0])  # sort by L

    # We'll find, for each j in [1..N], (min_cost[j], best_interval_id[j])
    INF = 10**18
    min_cost = [INF] * (N+1)
    best_i = [None] * (N+1)

    heap = []
    p = 0  # pointer for intervals array
    # intervals[p] = (L, R, C, i)

    for j in range(1, N+1):
        # push intervals that start at or before j
        while p < Q and intervals[p][0] == j:
            L, R, C, i_id = intervals[p]
            heapq.heappush(heap, (C, R, i_id))
            p += 1

        # pop intervals that no longer cover j (those with R < j)
        while heap and heap[0][1] < j:
            heapq.heappop(heap)

        # top of heap is the cheapest interval that covers j
        if heap:
            ctop, rtop, itop = heap[0]
            min_cost[j] = ctop
            best_i[j] = itop
        else:
            # no coverage for j => disconnected
            print(-1)
            return
    
    # Now, we do a DSU (Disjoint Set Union) over the N+Q vertices.
    # We'll label left-side vertices as 1..N, right-side as N+1..N+Q.

    parent = list(range(N+Q+1))
    rank = [0]*(N+Q+1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True
        return False

    # Step 2: For each j in 1..N, unite j with (N + best_i[j]) with cost = min_cost[j].
    # Accumulate that cost in MST sum.
    total_cost = 0

    for j in range(1, N+1):
        i_id = best_i[j]  # interval index
        cost_j = min_cost[j]
        # DS node for i_id is (N + i_id).
        # We'll add edge j -- (N+i_id).
        if union(j, N + i_id):
            total_cost += cost_j

    # Step 3: For each interval i in [1..Q], check if node (N + i) is connected.
    # If not, pick L_i (which we know is in the MST if it has coverage). 
    # Then union((N + i), L_i) with cost C_i if they are in different DS components.
    # This ensures the right-side vertex i is also in the MST.
    # Before that, we should confirm that L_i is not uncovered. If min_cost[L_i] == INF, then it's disconnected.

    # Let's store intervals in an array by their index i to quickly get (L_i, R_i, C_i).
    # We already have "intervals", but it's sorted by L, not by i. We'll build a lookup by i.

    # intervals list: (L, R, C, i_id) where i_id in [1..Q]
    # We want to index them by i_id. Let's build an array info[i_id] = (L, R, C).
    info = [None]*(Q+1)
    for L, R, C, i_id in intervals:
        info[i_id] = (L, R, C)

    for i_id in range(1, Q+1):
        node_right = N + i_id
        root_right = find(node_right)
        if root_right != find(1):
            # connect (N + i_id) with L_i if possible
            L_i, R_i, C_i = info[i_id]
            # If L_i was uncovered => disconnected
            if min_cost[L_i] == INF:
                print(-1)
                return
            if union(node_right, L_i):
                total_cost += C_i
    
    # Finally, check if the entire graph is in one component.
    # One quick way is to check if all vertices share the same root as 1 (or any node).
    root1 = find(1)
    for v in range(2, N+1):
        if find(v) != root1:
            print(-1)
            return
    for i_id in range(1, Q+1):
        if find(N + i_id) != root1:
            print(-1)
            return
    
    # If connected, print total_cost
    print(total_cost)

# Call solve() after defining it
solve()