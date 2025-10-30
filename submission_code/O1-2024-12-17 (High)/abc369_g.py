def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    edges = input_data[1:]
    # Adjacency list: for each node v, store list of (child, weight)
    # We'll treat 1 as root.  We only need to record undirected edges,
    # but we will do DFS with a "parent" parameter to distinguish children.

    adj = [[] for _ in range(N+1)]
    idx = 0
    for _ in range(N-1):
        u = int(edges[idx]); v = int(edges[idx+1]); w = int(edges[idx+2])
        idx += 3
        adj[u].append((v, w))
        adj[v].append((u, w))

    # We will do a DFS from node 1.  Define:
    #   dp[v] = an array (list) where dp[v][s] = maximum "coverage" of edges
    #            inside v's subtree if we "skip" exactly s leaves there.
    #   leaves_count[v] = number of (true) leaves in v's subtree (excluding v
    #                     if v=1 and has children; including v itself if it is
    #                     a leaf and v!=1).
    #
    # "Coverage" means the total sum of edge-weights in that subtree
    # that remain covered by at least one leaf that is NOT skipped.
    #
    # The tree is large, so we implement a "small-to-large" merging approach
    # to keep time roughly O(N log N).

    sys.setrecursionlimit(10**7)

    # We'll store dp[v] as a Python list (dynamic array).
    dp = [[] for _ in range(N+1)]
    # leaves_count[v] = number of leaves in v's subtree
    leaves_count = [0]*(N+1)
    visited = [False]*(N+1)

    # Merge function (small to large).
    # A, B are two dp arrays, with lengths lenA, lenB respectively.
    # We add edge_weight w for each combination (i, j) ONLY if j < lenB-1
    # i.e. if we do NOT skip all leaves in that child's subtree (meaning
    # we skip fewer than B's total leaves).
    # Return a new merged array newA of length (lenA + lenB - 1).
    def merge_dp(A, B, w):
        # Ensure A is the bigger one
        if len(A) < len(B):
            A, B = B, A
        lenA = len(A)
        lenB = len(B)
        newSize = lenA + lenB - 1
        # Use a large negative so we can do max() safely
        NEG = -10**18*5
        newA = [NEG]*newSize

        for i in range(lenA):
            ai = A[i]
            # i can go up to lenA-1
            for j in range(lenB):
                val = ai + B[j]
                # If j < (lenB-1), that means we're not skipping all leaves
                # in that child's subtree => add the edge w
                if j < lenB-1:
                    val += w
                # combine index is i+j
                if newA[i+j] < val:
                    newA[i+j] = val
        return newA

    # DFS function
    def dfs(v, parent):
        visited[v] = True
        children = []
        # collect children
        for (nx, w) in adj[v]:
            if nx == parent:
                continue
            dfs(nx, v)
            children.append(nx)

        # if v has no children (other than parent), and v!=1, v is a leaf
        if len(children) == 0 and v != 1:
            # leaf
            leaves_count[v] = 1
            # dp[v][0] = coverage if we skip 0 leaves? Actually we have two states:
            # skip=0 => keep this leaf => coverage=0
            # skip=1 => skip leaf => coverage=0
            dp[v] = [0, 0]
            return

        # not a leaf (or v=1 possibly)
        # start dp[v] with just [0], meaning no edges covered yet, no leaves
        dp[v] = [0]
        total_leaves = 0
        # merge each child's dp into dp[v]
        for c in children:
            w_c = 0
            # find the edge weight from v->c
            # we stored it in adj[v], it's w
            # but we do not want to loop again, let's just reuse the adjacency
            # Actually we have it from the for-loop, it's the same (nx, w).
            # We stored that in children as (nx).  We need the correct w though.
            # We'll retrieve it:
            for (xx, ww) in adj[v]:
                if xx == c:
                    w_c = ww
                    break

            dp[v] = merge_dp(dp[v], dp[c], w_c)
            total_leaves += leaves_count[c]

        leaves_count[v] = total_leaves
        # If v != 1 and has no children => it's a leaf (handled above).
        # If v=1 has children => not a leaf. If v!=1 but has children => also not a leaf.
        # So do nothing more.

    # Run DFS(1, -1)
    dfs(1, -1)

    # The number of leaves in the entire tree is leaves_count[1] (unless 1 is also
    # a leaf with no children, but that can't happen if N>=2).
    L = leaves_count[1]  
    # dp(1) has length L+1 now. dp(1)[s] = coverage if we skip s leaves in the entire tree.

    # For each K from 1..N:
    #   If K <= L => coverage = dp[1][L-K]
    #   else      => coverage = dp[1][0]   (picking more than L leaves is same as picking all L)
    # Then score = 2*coverage.

    res = []
    # dp[1] might have size L+1 (indices 0..L).
    # If L=0 (edge case?), that would mean no leaves except maybe the root?  For N>=2
    # that can't really happen, there's always at least 1 leaf.  But let's guard anyway.
    all_covered = dp[1][0] if len(dp[1])>0 else 0  # coverage if we skip 0 leaves
    for k in range(1, N+1):
        if k <= L:
            coverage = dp[1][L - k]
        else:
            coverage = all_covered
        # route length = 2 * coverage
        res.append(str(2*coverage))

    print("
".join(res))