def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    LRCS = input_data[2:]  # L_i, R_i, C_i in sequence

    # ---------------------------------------
    # 1) Read intervals and check coverage
    # ---------------------------------------
    intervals = []
    idx = 0
    for i in range(Q):
        L = int(LRCS[idx]); R = int(LRCS[idx+1]); C = int(LRCS[idx+2])
        idx += 3
        intervals.append((L, R, C, i))  # store also the index i
    # Quick check: If any old vertex is not in the union of intervals, graph is disconnected.
    # We can do a "line-sweep" style check that the union of [L_i,R_i] covers [1..N].
    # Sort by L ascending.
    intervals_by_left = sorted(intervals, key=lambda x: x[0])
    coverage_pos = 1
    current_end = 0
    i = 0
    while coverage_pos <= N:
        # If we have no interval starting at or before coverage_pos and current_end < coverage_pos => no coverage
        if i >= Q or intervals_by_left[i][0] > coverage_pos:
            # We can't advance current_end if we have no interval that begins on/before coverage_pos
            if current_end < coverage_pos:
                print(-1)
                return
        # While intervals begin on/before coverage_pos, extend current_end as much as possible
        while i < Q and intervals_by_left[i][0] <= coverage_pos:
            # intervals_by_left[i] = (L, R, C, idx)
            currR = intervals_by_left[i][1]
            if currR > current_end:
                current_end = currR
            i += 1
        # Now we can jump coverage_pos to current_end+1
        coverage_pos = current_end + 1

    # If we got here, all old vertices [1..N] are covered by the union of intervals
    # ---------------------------------------
    # 2) Build Kruskal-like MST using a "next array" trick
    #    Graph is bipartite: old vertices 1..N and new vertices N+1..N+Q.
    #    We will sort intervals by cost and try to connect each new vertex
    #    to old vertices in [L..R] in ascending cost order.
    # ---------------------------------------

    # Union-Find (Disjoint Set) implementation
    class UnionFind:
        __slots__ = ['parent','rank','count']
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0]*n
            self.count = n  # number of disjoint sets

        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            # union by rank
            if self.rank[rx] < self.rank[ry]:
                self.parent[rx] = ry
            elif self.rank[rx] > self.rank[ry]:
                self.parent[ry] = rx
            else:
                self.parent[ry] = rx
                self.rank[rx] += 1
            self.count -= 1
            return True

    # Sort the intervals by cost ascending
    intervals.sort(key=lambda x: x[2])  # (L,R,C,i) sort by C

    # Create union-find for all N+Q vertices
    # Old vertices: 0..(N-1)
    # New vertices: N..(N+Q-1)
    uf = UnionFind(N+Q)

    # "next" array for old vertices so we can skip those already processed
    # next[v] = v+1 initially, except for last which we point to N
    next_arr = list(range(1, N+1))  # 0-based: next_arr[v] = v+1
    next_arr[N-1] = N  # sentinel for end

    total_cost = 0
    edges_used = 0
    needed_edges = N + Q - 1  # MST needs exactly this many edges if connected

    for (L, R, cost, i_int) in intervals:
        new_vertex = N + i_int  # 0-based index for "new" vertex in UF
        # We'll attempt to connect all old vertices p in [L-1..R-1] with this new vertex
        p = L - 1  # 0-based
        while p < R:
            # If we've walked off the end (p == N), break
            if p >= N:
                break

            # Check if p is already connected with new_vertex
            rp = uf.find(p)
            rn = uf.find(new_vertex)
            if rp != rn:
                # they are in different components => unify
                uf.union(rp, rn)
                # only if union was successful (they were different)
                total_cost += cost
                edges_used += 1
                if edges_used == needed_edges:
                    # MST complete
                    print(total_cost)
                    return

            # Move p forward using next-array
            old_p = p
            p = next_arr[p]
            # "Shortcut" so we skip old_p in future
            next_arr[old_p] = p

    # If we exit the loop without having edges_used == needed_edges,
    # check if the entire graph is indeed connected
    # We can check uf.count or edges_used
    if edges_used < needed_edges:
        # Not enough edges to connect all
        print(-1)
    else:
        print(total_cost)

# Call main() at the end
if __name__ == "__main__":
    main()