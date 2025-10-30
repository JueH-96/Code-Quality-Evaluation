def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Build the tree: adjacency list and degree count.
    # We use 0-indexing internally.
    adj = [[] for _ in range(n)]
    deg = [0] * n
    for _ in range(n - 1):
        u = int(next(it)) - 1
        v = int(next(it)) - 1
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # Idea:
    # A "Snowflake Tree" produced by the procedure has three layers:
    #   • Center (red): 1 vertex.
    #   • Blue vertices directly attached to the center. The procedure adds x of these.
    #   • For each blue vertex, attach y leaves.
    # So the total vertices is 1 + x + x*y = 1 + x*(1+y) and note that x,y are positive.
    #
    # In any candidate induced subgraph S that is isomorphic to a snowflake tree, one vertex (the center)
    # will have blue vertices chosen among its neighbors. For each blue vertex u (which must be adjacent to the center v),
    # we plan to “use” exactly y of its other neighbors – these will serve as its leaves.
    # Since we are allowed to delete vertices, we may keep only a subset of neighbors for every vertex.
    #
    # So if we fix a candidate center v in T, then we may choose among v’s neighbors which ones will be blue.
    # But note:
    #   • We must have at least one blue vertex (x > 0).
    #   • For a neighbor u of v to be chosen as blue, u must be able to “offer” at least one leaf.
    #     (That is, u must have at least one neighbor other than v; equivalently, deg(u) >= 2.)
    #   • Given a blue candidate u, the maximum number of leaves we could “attach” (keep in S) is (deg(u) - 1)
    #     because one neighbor (v) is already consumed.
    #
    # Given a candidate center v, let’s consider the set of its neighbors (blue candidates) u for which deg(u) >= 2.
    # For such a candidate u the number of “available” leaves is L[u] = deg(u) - 1.
    # If we choose a set B of blue candidates from v then we must choose a common y (a positive integer)
    # with y <= min_{u in B} L[u]. (Remember that in the procedure each blue vertex gets exactly y leaves.)
    #
    # So if we choose k blue vertices from v (say, the ones with highest L[u] values),
    # we can achieve a snowflake with size:
    #   1 (for center) + k (blue vertices) + k*y, and we want to choose y as large as possible,
    # namely y = min_{u in chosen blue candidates} (deg(u)-1).
    # In other words, for a candidate center v if we sort the available-leaf counts for neighbors that are eligible (deg>=2)
    # in descending order, then for each possible k (1-based) if the kth highest value is L,
    # we could achieve a snowflake S of size = 1 + k*(1 + L).
    # We do this for every vertex v as center and take the maximum size found.
    #
    # Finally, since our goal is to delete as few vertices as possible,
    # we want to keep as many vertices as possible in an induced snowflake subgraph.
    # Answer = n - (maximum number of vertices in a snowflake subgraph S).
    
    global_best = 0
    # Try every vertex as potential center
    for v in range(n):
        cand = []
        for nb in adj[v]:
            # A neighbor can serve as blue vertex only if it has at least one extra neighbor.
            if deg[nb] >= 2:
                cand.append(deg[nb] - 1)  # available leaves from this blue candidate
        if not cand:
            continue  # v cannot be center because we need at least one blue vertex
        # Sort the available counts in descending order
        cand.sort(reverse=True)
        best_here = 0
        # When we choose the top k candidates, the common y we may use is the kth value.
        # The snowflake would then have size = 1 (center) + k (blue) + k * (candidate[k-1]) (leaves).
        k = 0
        for a in cand:
            k += 1
            size = 1 + k * (1 + a)
            if size > best_here:
                best_here = size
        if best_here > global_best:
            global_best = best_here

    # Because our snowflake S must be an induced subgraph of T,
    # we can keep at most n vertices. (It is guaranteed that a snowflake is possible.)
    if global_best > n:
        global_best = n
    ans = n - global_best
    
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()