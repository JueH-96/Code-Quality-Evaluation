def solve():
    import sys
    sys.setrecursionlimit(10**7)

    input = sys.stdin.readline
    N = int(input())
    adj = [[] for _ in range(N+1)]
    for _ in range(N-1):
        u,v,l = map(int,input().split())
        adj[u].append((v,l))
        adj[v].append((u,l))

    # ----------------------------------------------------------------
    # Overview of the solution:
    #
    # We want, for each K = 1..N, the maximum possible "subtree coverage"
    # when Aoki chooses K distinct vertices and Takahashi must visit them
    # in a walk starting and ending at vertex 1. The length of that walk
    # in a tree is 2 * (sum of edges in the minimal subtree that contains
    # vertex 1 and those K chosen vertices). Hence if we let "coverage(X)"
    # be the sum of (unique) edge lengths in the union of paths from 1 to
    # all vertices in X, then the score is 2 * coverage(X). Aoki wants
    # to maximize coverage(X) by choosing X of size K.
    #
    # Observe that in a tree (rooted at 1), the minimal subtree connecting
    # 1 and a chosen vertex x is simply the unique path 1->x. The union of
    # those paths for X is all edges that lie on at least one path 1->x
    # with x in X.
    #
    # Hence the problem becomes:
    #   "Pick K distinct vertices (other than 1, picking 1 does not add edges)
    #    so that the union of their root->vertex paths covers the maximum sum
    #    of edge lengths."
    #
    # A standard tree-DP method can be used to compute, for each subtree,
    # how the coverage grows as we pick various numbers of nodes from that
    # subtree.  Then we combine children in a knap-sack style, adding the
    # parent->child edge length if we pick >0 nodes in that child subtree.
    #
    # Finally, the answer for each K is 2 * dp[1][K], where dp[1][k] is the
    # maximum coverage inside the entire tree (rooted at 1) when exactly k
    # (distinct) vertices (excluding the root itself) are chosen.
    #
    # Note that we must do this efficiently (N up to 2e5). The classic way
    # is a "merge small-to-large" technique, storing dp[u] in a list of length
    # up to the size of u's subtree, and merging children’s dp-lists in
    # descending order of size.  This guarantees overall O(N log N) or O(N)
    # complexity (amortized).  The final dp[1][k] for k=1..N (or up to N-1
    # since we can't pick the root) gives us the coverage for each K; but
    # we do want K up to N, and picking "the root" adds no coverage, so
    # dp[1][N] = dp[1][N-1] in effect.  We will store up to dp[1][N-1].
    #
    # Implementation details:
    # 1) Root the tree at 1.
    # 2) Do a DFS to build dp[u], where dp[u][k] = maximum coverage in subtree u
    #    if exactly k vertices (excluding u) are chosen there.
    #    - dp[u][0] = 0 always (no chosen nodes => no edges used).
    #    - To merge a child c having dp[c], for "x+y" picks we do:
    #         newdp[x+y] = max of
    #            newdp[x+y],
    #            dp[u][x] + dp[c][y] + ( (y>0) ? edge_length(u->c) : 0 )
    #    - We'll do a small-to-large merge to stay efficient.
    #
    # 3) After processing children of 1, dp[1][k] is the coverage for picking k
    #    nodes below 1.  The result for each K is 2*dp[1][K], and for K> the
    #    subtree size, it remains the same as dp[1][subtree_size(1)-1].
    #
    # 4) Print answers for K = 1..N.  (If K exceeds the number of nodes below
    #    the root, the coverage can't grow further, so it stays at dp[1][N-1].)
    #
    # Let's implement.
    # ----------------------------------------------------------------

    sys.setrecursionlimit(10**7)

    # Build tree (already in adj).
    # We'll root at 1, compute subtree sizes, then do a DFS with "small-to-large" merges of dp.

    # dp[u] will be a list: dp[u][k] = maximum coverage if we pick exactly k nodes in subtree of u (excluding u)
    # We'll store dp[u] in a python list that can grow.  We'll do small-to-large merges.

    # We also need subtree_size[u] to know how large dp[u] can become.
    subtree_size = [0]*(N+1)
    visited = [False]*(N+1)
    parent = [-1]*(N+1)
    # adjacency: adj[u] = [(v1,len1), (v2,len2), ...]

    # First pass: compute a tree DFS from 1, compute subtree sizes and parents.
    def dfs_size(u):
        visited[u] = True
        sz = 1
        for (w,_) in adj[u]:
            if not visited[w]:
                parent[w] = u
                sz += dfs_size(w)
        subtree_size[u] = sz
        return sz

    dfs_size(1)

    # Prepare containers for dp
    dp = [[] for _ in range(N+1)]  # dp[u] a list of length subtree_size[u], indexed by k from 0..(subtree_size[u]-1)
    # We'll also store children for a second DFS pass
    children = [[] for _ in range(N+1)]
    for u in range(2, N+1):
        p = parent[u]
        w = 0
        for (c,l) in adj[u]:
            if c == p:
                w = l
                break
        # w is the length of edge p->u. We'll do it from parent's side.
    for u in range(1, N+1):
        for (v,l) in adj[u]:
            if parent[v] == u:  # v is child of u
                children[u].append((v,l))

    # We'll define a DFS that computes dp[u] from children[u].
    # We'll do a post-order approach. Then we'll do small-to-large merges
    # for the children of u.

    def dfs_dp(u):
        # start dp[u] = [0] * subtree_size[u]
        # but we don't know subtree_size[u] yet... we do have it from dfs_size
        dp[u] = [0]*(subtree_size[u])  # index: 0..(subtree_size[u]-1)
        # dp[u][0] = 0 by default
        size_now = 0  # how many picks we have accounted for so far

        for (c, elen) in children[u]:
            dfs_dp(c)
            # Merge dp[c] into dp[u]
            # let dp[c] have length = subtree_size[c]
            child_len = subtree_size[c]
            # We'll do a small-to-large merge:
            new_size = size_now + child_len
            newdp = [-1]*(new_size+1)
            for take_u in range(size_now+1):
                base_val = dp[u][take_u]
                if base_val < 0:
                    continue
                # picking 0 from c:
                # coverage doesn't increase by elen, because we didn't pick anything from c
                nxt_idx = take_u
                if newdp[nxt_idx] < base_val:
                    newdp[nxt_idx] = base_val
                # picking y>0 from c => we add elen + dp[c][y]
                for y in range(1, child_len+1):
                    if dp[c][y-1] < 0:
                        # might be that we haven't properly init dp[c] with -1, let's fix that
                        continue
                    val_c = dp[c][y-1]
                    candidate = base_val + val_c + elen
                    nxt_idx = take_u + y
                    if newdp[nxt_idx] < candidate:
                        newdp[nxt_idx] = candidate
            # now prune newdp if it didn't get updated
            dp[u] = newdp
            size_now = new_size
        # fill any leftover -1 with 0 if k=0 or remain -1 else
        # But we do want dp[u][k] = -1 for impossible. It's possible that
        # we don't pick more than size_now anyway.  We'll keep them as is.
        # dp[u] now has length size_now+1
        # But subtree_size[u] might be bigger than size_now+1 if we count
        # "excluding u". Actually size_now = sum of subtree_size[c] for children c
        # so size_now = (subtree_size[u] - 1). That matches the length we want.
        # We'll keep dp[u] as is.

    # However, we must initialize dp[c][k] for c "leaf"?  By default we made dp[c] all zeros?
    # That means dp[c][k] only meaningful if k=0.  We need "impossible" markers for k>0.
    # But we do want dp[c][1] = 0 for a leaf? Actually picking 1 node "in the subtree of c
    # excluding c itself" is impossible if c is a leaf and has no children. So dp[c][1] = -∞.
    # Let's fix initialization for all dp arrays with -∞, dp[u][0]=0.

    # We'll do that after computing subtree_size but before the DFS:
    for u in range(1, N+1):
        dp[u] = [-1]*(subtree_size[u])
        dp[u][0] = 0

    # Now define the DFS that merges children:
    def dfs_dp(u):
        size_now = 0  # how many picks accounted for among children
        for (c, elen) in children[u]:
            dfs_dp(c)
            child_len = subtree_size[c]
            new_size = size_now + child_len
            newdp = [-1]*(new_size+1)
            for x in range(size_now+1):
                base_val = dp[u][x]
                if base_val < 0:
                    continue
                # 0 picks from c => no edge to c
                if newdp[x] < base_val:
                    newdp[x] = base_val
                # 1..child_len picks from c => add edge length + dp[c][y]
                for y in range(1, child_len+1):
                    val_c = dp[c][y-1]
                    if val_c < 0:
                        continue
                    cand = base_val + val_c + elen
                    if newdp[x+y] < cand:
                        newdp[x+y] = cand
            dp[u] = newdp
            size_now = new_size

    # Compute dp arrays
    dfs_dp(1)

    # dp[1][k] is coverage for exactly k picks among children of 1 (subtree(1) minus itself).
    # The subtree_size(1) = N, so dp[1] has length N. dp[1][k] for k>0 might be -1 if it was impossible
    # but we expect dp[1][k] for k up to (N-1) to be valid. In a tree, we can pick up to N-1 nodes (excluding root).
    # If K > N-1, the coverage won't grow further, so we can treat dp[1][K] = dp[1][N-1] for K>=N.

    # Now gather results: for K=1..N, answer = 2*dp[1][K], with dp[1][K] = dp[1][N-1] if K> N-1.
    # But we must handle the possibility that some dp[1][k] is -1 (which theoretically shouldn't
    # happen if the subtree is connected, but just in case, we'll keep track of the maximum feasible).
    max_cov = dp[1]
    # fix any -1 by carrying forward the last non-negative
    for i in range(1, N):
        if max_cov[i] < max_cov[i-1]:
            max_cov[i] = max_cov[i-1]

    # print results
    import sys
    out = []
    for K in range(1, N+1):
        useK = min(K, N-1)  # we can pick at most N-1 distinct vertices excluding 1
        cov = max_cov[useK]
        ans = 2 * cov
        out.append(str(ans))
    print("
".join(out))