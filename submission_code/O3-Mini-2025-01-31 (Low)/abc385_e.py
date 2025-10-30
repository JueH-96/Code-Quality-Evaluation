def main():
    import sys
    import bisect
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Build tree representation and count vertex degrees.
    deg = [0] * (n+1)
    adj = [[] for _ in range(n+1)]
    for _ in range(n-1):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
        deg[u] += 1
        deg[v] += 1

    # Explanation of the approach:
    #
    # A "Snowflake Tree" is produced by:
    #   • a red center vertex,
    #   • x blue vertices attached to the center, and
    #   • for each blue vertex, exactly y leaf vertices attached.
    #
    # In the final induced structure, the vertex counts are:
    #     1 (red) + x*(1 + y)  (each branch contributes its blue vertex plus y leaves)
    #
    # Our allowed operation is deletion of any vertices and we want to keep as many vertices as possible.
    # Equivalently, we want to find an induced subgraph of T that is isomorphic to a snowflake tree
    # (with some positive integers x and y) and then the answer is N minus the maximum size of such a structure.
    #
    # How to check if a chosen vertex v (as red) can serve as the center?
    #   For any candidate blue vertex u (neighbor of v) we plan to keep, we know that in the final structure u 
    #   must have exactly one edge kept to the red v and then exactly y leaves attached – a total degree of y+1.
    #   In the original tree u has deg(u) and we are allowed to delete extra neighbors.
    #   The potential to serve as a blue branch is available if u has at least y neighbors (that is, deg(u)-1>=y).
    #
    # For each candidate red vertex v:
    #   • Look at all its neighbors u. The value available at u is (deg(u) - 1) (number of neighbors 
    #     other than v that we could keep as leaves).
    #   • For a fixed y (which must be at least 1) and must be the same for all branches, the neighbor u qualifies if
    #         deg(u)-1 >= y.
    #   • Let count be the number of neighbors of v that qualify.
    #   • Then, if we take all these green branches we get a snowflake tree with:
    #           Total vertices = 1 (red) + count * (1 blue + y leaves) = 1 + (y+1)*count.
    #
    # We try all vertices v (as red center) and for each v, consider all candidate y from 1 up to max{ deg(u)-1 among u in N(v) }.
    # For each chosen v and y (provided there is at least one blue branch) we compute
    #           S = 1 + (y+1)*count
    # and keep track of the maximum S among all such valid snowflake trees.
    #
    # Our answer is then: minimum vertices to delete = n - (maximum size of a snowflake tree found).
    #
    # Complexity:
    #   Summing over each vertex v, the work done is proportional to deg(v) plus iterating over possible y.
    #   In the worst‐case (for example a star) this works within the constraints.
    
    best = 0
    # Test each vertex v as potential red center.
    for v in range(1, n+1):
        if not adj[v]:
            continue
        # Build a sorted list "lst" containing the value (deg(u) - 1) for each neighbor u of v.
        lst = []
        for u in adj[v]:
            lst.append(deg[u] - 1)
        lst.sort()
        if not lst:
            continue
        max_available = lst[-1]
        # For each potential y value (note: y must be at least 1)
        for y in range(1, max_available + 1):
            # Count how many neighbors have available leaves >= y.
            idx = bisect.bisect_left(lst, y)
            count = len(lst) - idx
            if count < 1:
                continue  # must have at least one blue branch
            candidate_size = 1 + (y + 1) * count  # 1 red + count branches (each gives blue+ y leaves)
            if candidate_size > best:
                best = candidate_size
    if best > n:
        best = n
    # The answer is the number of vertices to delete.
    print(n - best)

if __name__ == "__main__":
    main()