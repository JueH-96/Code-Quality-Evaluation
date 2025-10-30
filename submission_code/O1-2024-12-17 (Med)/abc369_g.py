def main():
    import sys
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    N = int(input().strip())
    # Adjacency list: edges[u] = list of (v, length)
    edges = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u,v,l = map(int, input().split())
        edges[u].append((v,l))
        edges[v].append((u,l))

    # We will do a DFS rooted at 1 to:
    # 1) find parent[] and subtree sizes
    # 2) compute a DP array dp[v], where dp[v][k] = maximum "coverage inside v's subtree" if we pick k nodes there
    #
    # The "coverage inside v's subtree" does NOT include the edge from the parent(v)->v (that edge is accounted
    # for at the parent's level if we decide to pick >0 nodes in v's subtree).
    #
    # Finally, dp[1][k] will be the maximum coverage if we pick k nodes in the entire tree (root=1).  The required
    # game score is 2 * dp[1][k], for k = 1..N.
    #
    # Implementation detail:
    # - We do a post-order DFS that returns dp[v].
    # - For a leaf v, dp[v] = [0,0], meaning coverage(0 picks)=0, coverage(1 pick)=0 (no child-edges).
    # - For an internal node v, we:
    #     - identify a "heavy child" cH (largest subtree) to process first
    #     - take dp[cH] as a base array and add length(v->cH) to all entries dp[cH][x] for x>0
    #     - then for each other child c, compute dp[c], build an array mg = [ dp[c][x]+(x>0? length(v->c) : 0 ) ]
    #       and "merge" mg into the dp[v]-array using small-to-large merging.
    #     - at the end we allow "picking v itself" to bump up the index by 1 without adding coverage (so we do
    #       a final pass that does dpv[i+1] = max(dpv[i+1], dpv[i]) so that dpv grows by one more index).
    # - The length of dp[v] will become subtree_size(v)+1.  dp[v][k] is the best coverage for picking k nodes
    #   in that subtree (they may be anywhere below v, or v itself).
    #
    # - In the end dp[1] has length N+1, so dp[1][k] is valid for k=0..N.  We print 2*dp[1][k] for k in 1..N.

    parent = [-1]*(N+1)
    sz = [0]*(N+1)
    visited = [False]*(N+1)

    def dfs_sz(v):
        visited[v] = True
        sz[v] = 1
        for (c,w) in edges[v]:
            if not visited[c]:
                parent[c] = v
                dfs_sz(c)
                sz[v] += sz[c]

    # compute parents & subtree sizes from root=1
    parent[1] = 0
    dfs_sz(1)

    # dp[v] will be a list of length up to sz[v]+1
    dp = [[] for _ in range(N+1)]

    # A helper to merge two dp-arrays (A,B) in "knapsack" style:
    #   new[i+j] = max( new[i+j], A[i] + B[j] )
    # We do "small-to-large" merging to keep complexity near O(N log N) overall.
    def merge_arrays(A, B):
        # Ensure A is the bigger array (so we merge B into A) if possible
        if len(A) < len(B):
            A, B = B, A
        # Now len(A) >= len(B)
        new_len = len(A) + len(B) - 1
        newA = [-1]*(new_len)
        for i in range(len(A)):
            valA = A[i]
            if valA < 0:
                continue
            for j in range(len(B)):
                if B[j] < 0:
                    continue
                idx = i + j
                cand = valA + B[j]
                if cand > newA[idx]:
                    newA[idx] = cand
        return newA

    # Post-order DFS to fill dp[v]
    def dfs_dp(v, p):
        # find the "heavy child" = child with largest sz[c]
        heavy = -1
        heavy_sz = 0
        heavy_w = 0  # edge-length to that heavy child
        for (c,w) in edges[v]:
            if c == p: 
                continue
            if sz[c] > heavy_sz:
                heavy_sz = sz[c]
                heavy = c
                heavy_w = w

        # if no children => leaf
        if heavy == -1:
            # dp[v] = coverage for 0 picks => 0, 1 pick => 0 (no edges in child's subtree)
            dp[v] = [0, 0]
            return

        # Recurse on heavy child first
        dfs_dp(heavy, v)
        # Start dp[v] as a copy of dp[heavy]
        dpv = dp[heavy][:]  # copy
        # Add the edge (v->heavy) cost for all picks > 0
        for i in range(1, len(dpv)):
            dpv[i] += heavy_w

        # Now merge in the other children
        for (c, w) in edges[v]:
            if c == p or c == heavy:
                continue
            dfs_dp(c, v)
            # Build mgc array for child c: mgc[i] = dp[c][i] + (i>0 ? w : 0)
            # (the coverage inside c's subtree plus the parent's edge cost if we pick >=1 there)
            child_dp = dp[c]
            mgc = [0]*len(child_dp)
            for i in range(1, len(child_dp)):
                mgc[i] = child_dp[i] + w
            
            # Merge mgc into dpv
            dpv = merge_arrays(dpv, mgc)

        # Finally, allow "picking v itself" which uses up 1 more pick but does not add coverage:
        # dpv[i+1] = max(dpv[i+1], dpv[i]) in descending order
        dpv.append(dpv[-1])  # extend by 1
        for i in range(len(dpv)-2, -1, -1):
            if dpv[i] > dpv[i+1]:
                dpv[i+1] = dpv[i]

        dp[v] = dpv

    dfs_dp(1, 0)

    # dp[1] has length = sz[1]+1 = N+1
    # dp[1][k] is the max coverage for k picks in the entire tree
    # answer for each k is 2 * dp[1][k]
    # careful with indexing: dp[1][0], dp[1][1], ...
    ans_dp = dp[1]
    for k in range(1, N+1):
        print(2 * ans_dp[k])