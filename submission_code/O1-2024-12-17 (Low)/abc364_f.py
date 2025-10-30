def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    LRC = input_data[2:]
    
    # ----------------------------------------------------------------------------
    # We have a bipartite graph:
    #   • "First partition": vertices 1..N  (we'll treat them as 0..N-1 in 0-based).
    #   • "Second partition": vertices N+1..N+Q (we'll treat them as N..N+Q-1 in 0-based).
    #
    # For i = 1..Q (0-based: i = 0..Q-1), we add edges of cost C_i from vertex (N + i)
    # in the second partition to all vertices j in [L_i..R_i] in the first partition.
    #
    # We want to:
    #   1) Check if the final graph is connected (i.e. has a single connected component).
    #   2) If connected, find the cost of the MST.
    #
    # Directly listing all edges can be O(N*Q), which is too large (up to 4e10).
    # Instead, note the graph is bipartite and each new vertex i (in second partition)
    # connects to an interval [L_i..R_i] in the first partition (1-based).
    #
    # We will sort all Q "interval-vertices" (i.e. second-partition vertices) by cost C_i.
    # Then we process them in ascending cost. When processing an interval i with cost c,
    # we effectively want to "union" the node (N+i) with all covered nodes j in [L_i..R_i].
    #
    # However, we cannot union them one-by-one if the interval is large. Instead we use
    # an interval-merging data structure for the first partition (1..N). Whenever we
    # connect an interval [L..R], we unify (N+i) with every distinct connected component
    # in the first partition that has non-empty intersection with [L..R]. Then those
    # components merge into one bigger interval (since they all get connected via (N+i)).
    #
    # We'll maintain a set (or balanced tree) of disjoint intervals covering [1..N]
    # that represent the connected components in the first partition that have been
    # formed so far (in the MST sense). Initially, no edges exist, so each of 1..N is
    # its own separate interval [k..k]. As we process edges in ascending cost order,
    # whenever an interval i = [L_i..R_i] is processed, we find all existing intervals
    # that intersect [L_i..R_i], unify them (and unify them with node N+i), and merge
    # them into one big interval in the data structure.
    #
    # Each time we perform a union operation that actually merges two different DS
    # components, we add C_i to the MST cost. After processing all intervals, we check
    # if we ended up with a single DS component (i.e. all N+Q vertices are in one set).
    #
    # If yes, we output the total MST cost; otherwise, -1.
    #
    # Complexity: O((N+Q) log(N+Q)) with a careful interval-merging approach.
    # ----------------------------------------------------------------------------

    # Union-Find (Disjoint Set) for all N+Q vertices.
    # We'll 0-base the first partition as [0..N-1], second partition as [N..N+Q-1].
    parent = list(range(N + Q))
    size = [1] * (N + Q)
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        return True

    # Read the intervals [L_i..R_i] (1-based) along with cost C_i.
    intervals = []
    idx = 0
    for i in range(Q):
        l = int(LRC[idx]); r = int(LRC[idx+1]); c = int(LRC[idx+2])
        idx += 3
        # Store them with 0-based for the first partition: [l-1..r-1].
        intervals.append((l-1, r-1, c, i))  # (L, R, cost, i)

    # Sort intervals by cost ascending
    intervals.sort(key=lambda x: x[2])

    # We'll maintain a balanced structure of "active intervals" in [0..N-1].
    # Each is represented as (start, end) disjoint, covering only the
    # portion that is in one DS component. We'll keep them in a sorted list
    # keyed by 'start'. We also store "ds_rep" for each such component,
    # but we can deduce a representative from any point in the interval,
    # so let's store the left endpoint as a canonical representative
    # or store a separate "rep" array. For simplicity, we'll store "start, end".
    #
    # We need to do lookups: find all intervals that intersect [L..R]. Then
    # merge them with [L..R] into one bigger interval. That means removing them
    # all from the structure and re-inserting the union's bounding [newL..newR].
    #
    # Implementation detail in Python:
    # We'll store intervals in a "list of (start, end)" in ascending order,
    # plus a parallel list for binary-search. We'll do merges with a while-loop.
    # Each time we might remove a few intervals. The total complexity is O((N+Q) log N).
    
    import bisect

    active_intervals = []
    # Initially, each integer 0..N-1 is its own interval.
    # But storing each as [k..k] for k in [0..N-1] would be huge if N=200000.
    # Instead, note that initially none of the first-partition vertices are
    # "connected among themselves" by any edge, so they indeed are separate components.
    #
    # We do have to have them all in the data structure so we can find them. That
    # is potentially large, but we can handle it with O(N) space. We'll store them as
    # intervals = [(0,0), (1,1), ..., (N-1,N-1)] and a sorted array of starts. OK.
    
    # Build them:
    active_intervals = [(i, i) for i in range(N)]  # each single node is its own interval
    starts = list(range(N))  # the start indices, sorted
    
    # We'll define helper functions to find intervals by position x.
    def find_interval(x):
        # We want to find the rightmost interval whose start <= x (if any).
        # We'll do a bisect_right(starts, x) - 1.
        i = bisect.bisect_right(starts, x) - 1
        if i < 0:
            return -1
        # Check if active_intervals[i] intersects x
        (s, e) = active_intervals[i]
        if s <= x <= e:
            return i
        return -1

    # We'll define a routine that merges all intervals that intersect [L..R].
    # It returns the final bounding interval [mergedL..mergedR], plus
    # a list of distinct representatives that got merged (so we can union them
    # with the second-partition node).
    def merge_intervals(L, R):
        """
        Find all intervals that intersect [L..R], remove them, and merge them into
        the union bounding interval [newL..newR]. Also collect one representative
        from each interval (say the left endpoint).
        Return (newL, newR, reps).
        If no interval intersects, return None.
        """
        # We'll do a loop, starting from i = find_interval(R) (the rightmost possible),
        # then move leftwards while intervals intersect [L..R]. Or we can start from
        # find_interval(L) and move rightwards. Let's do rightwards approach:
        i = find_interval(L)
        if i < 0:
            # no interval starts <= L => maybe they start > L but still overlap?
            # better check the next interval if any
            j = bisect.bisect_left(starts, L)
            if j >= len(starts):
                return None  # no intervals at all
            # check if that interval intersects [L..R]
            s, e = active_intervals[j]
            if s > R:
                return None
            i = j

        merged_left = None
        merged_right = None
        reps = []
        changed_any = False

        # Because merging might skip forward, we'll do a while loop
        while True:
            if i >= len(active_intervals):
                break
            s, e = active_intervals[i]
            if s > R:
                # no more intersection
                break
            # Now, this interval [s..e] intersects [L..R] if e >= L.
            # We do indeed have e >= s > R => no. We must check if e < L => no intersection.
            if e < L:
                i += 1
                continue
            # So it does intersect [L..R].
            changed_any = True
            # Merge it into our bounding interval.
            if merged_left is None:
                merged_left = min(s, L)
                merged_right = max(e, R)
            else:
                if s < merged_left:
                    merged_left = s
                if e > merged_right:
                    merged_right = e
            # store a representative (take s, for instance)
            reps.append(s)
            # remove this interval from active_intervals
            del active_intervals[i]
            del starts[i]
            # no i += 1 here because we've removed the interval from the list
            # so the next interval to examine is still at index i
            # but we might need to continue merging more intervals as long
            # as there's overlap with [L..R], or with [merged_left..merged_right].
            # Actually, we intersect [L..R], but once we've started merging,
            # effectively we must consider overlap with the *expanded* interval
            # [merged_left..merged_right], because we unify them all.
            # So let's update L,R = merged_left, merged_right,
            # and keep merging as far as there's intersection.
            L, R = merged_left, merged_right
        if not changed_any:
            return None
        # Now we might need to keep merging any intervals that start <= merged_right
        # even if they start after we started. So let's do a while loop that merges
        # newly intersecting intervals with the updated bounding region.
        # We'll keep going until no new merges happen.
        while True:
            # find the first interval that might overlap new [L..R]
            j = bisect.bisect_left(starts, L)
            if j > 0:
                j -= 1
            if j < 0:
                j = 0
            new_merge_happened = False
            while j < len(active_intervals):
                s, e = active_intervals[j]
                if s > R:
                    break
                if e < L:
                    j += 1
                    continue
                # Overlaps
                new_merge_happened = True
                # Merge
                if s < merged_left:
                    merged_left = s
                if e > merged_right:
                    merged_right = e
                reps.append(s)
                del active_intervals[j]
                del starts[j]
                L, R = merged_left, merged_right
                # no j += 1
            if not new_merge_happened:
                break

        return (merged_left, merged_right, reps)

    total_edges_used = 0
    mst_cost = 0

    # To check connectivity in the end, we need exactly (N+Q-1) edges in the MST
    # if the graph is connected. We'll gather them by union operations.
    # We'll process intervals in ascending cost order:
    for (L_i, R_i, c, i_orig) in intervals:
        # second-partition node is N + i_orig (0-based).
        node_sp = N + i_orig

        # Repeatedly merge any intervals that intersect [L_i..R_i].
        # If none exist, that means these positions in [L_i..R_i] might
        # already have been merged into a bigger interval that lies entirely
        # outside or inside. Let's call the merge function once, then keep
        # calling it if we produce a bigger bounding interval.
        merged = merge_intervals(L_i, R_i)
        if merged is None:
            # That means there was no uncovered portion in [L_i..R_i].
            # Possibly everything in [L_i..R_i] is already part of some intervals
            # that do not overlap. But let's check if there's an interval that
            # does overlap if we do a direct search. If truly there is no interval,
            # it means [L_i..R_i] might have been fully absorbed by some
            # previously merged big interval that does not physically intersect.
            #
            # Let's do an interval lookup for any single point in [L_i..R_i].
            # If the set is empty or there's no interval that intersects,
            # those nodes might have been merged with intervals far away.
            # Actually, it means there's no separate "active interval" to unify here,
            # so no new edges will be added. We move on.
            continue
        else:
            # We got (newL, newR, reps). Now reps[] are the distinct intervals' reps.
            newL, newR, reps = merged
            # All these reps were separate intervals in the first partition. Now
            # we unify them among themselves via "node_sp" (the second-part node).
            # So effectively, we do DS-union(node_sp, rep) for each rep.
            # Each time it merges, we add cost c.
            # Then we insert the merged interval [newL, newR] back into active_intervals.
            # But before that, we can unify them in DS first.
            # We'll choose the first rep as "canonical" for the new interval, or newL.
            # But to ensure correctness, let's unify each rep with node_sp.

            root_already = None
            for rp in reps:
                if union(node_sp, rp):
                    # If union is new, we add cost
                    mst_cost += c
                    total_edges_used += 1
            
            # Now we insert [newL, newR] back into the data structure as a new interval.
            # (We store it at an appropriate position in sorted order.)
            pos = bisect.bisect_left(starts, newL)
            active_intervals.insert(pos, (newL, newR))
            starts.insert(pos, newL)

    # After processing all intervals, we check if the graph is connected:
    # If it is connected, we should have used exactly (N+Q-1) edges in MST
    # (since a spanning tree on (N+Q) vertices has exactly (N+Q-1) edges).
    # Also we can check if union-find has exactly one unique representative,
    # but counting edges is enough. If total_edges_used != (N+Q-1), it's not connected.
    
    if total_edges_used == (N + Q - 1):
        # We still need to check if indeed everything is in one component
        # because in degenerate cases it might unify some cycles. But generally,
        # if a spanning tree on (N+Q) nodes uses (N+Q-1) edges, it must be connected.
        # We'll do a final check anyway:
        # find rep of 0, and confirm all have the same rep:
        r0 = find(0)
        for v in range(1, N+Q):
            if find(v) != r0:
                print(-1)
                return
        print(mst_cost)
    else:
        print(-1)

# Do not forget to call main!
if __name__ == "__main__":
    main()