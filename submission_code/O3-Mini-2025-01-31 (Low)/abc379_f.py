def main():
    import sys
    import math
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    H = [int(next(it)) for _ in range(N)]
    # For 1-indexed buildings (positions 1..N) we use 0-index internally.

    # (1) Compute next greater element for each building.
    # For a building i (0-indexed) define f[i] = smallest index j > i with H[j] > H[i],
    # which is exactly the next building that “erupts” in height – this exactly forms the chain
    # of visible buildings (from a viewpoint, the visible chain is: f[i], f[f[i]], f[f[f[i]]], …).
    f = [-1] * N
    stack = []
    for i in range(N):
        # while current building is taller than building at the top of stack:
        while stack and H[stack[-1]] < H[i]:
            f[stack.pop()] = i
        stack.append(i)
    # f[i] is -1 if no greater building to its east.

    # (2) Build a “next greater tree” where each node i (except if f[i] = -1) has parent f[i].
    parent = f[:]  # parent's pointer: for i, parent[i] is the next greater building that is visible from i.
    # Every building’s chain (the visible buildings) is: get_ancestor( i, 1 ), get_ancestor(i,2), etc.
    # We now compute the "depth" for each building in this “parent” chain.
    depth = [0]*N
    # Notice that since f[i] > i always (if exists), we can compute depth in reverse order.
    for i in range(N-1, -1, -1):
        if parent[i] == -1:
            depth[i] = 0
        else:
            depth[i] = depth[parent[i]] + 1

    # (3) Precompute binary lifting table – standard procedure for fast LCA queries.
    LOG = math.ceil(math.log2(N+1))
    up = [[-1]*N for _ in range(LOG)]
    for i in range(N):
        up[0][i] = parent[i]
    for j in range(1, LOG):
        for i in range(N):
            if up[j-1][i] != -1:
                up[j][i] = up[j-1][up[j-1][i]]
            else:
                up[j][i] = -1

    # get_ancestor(v, k) returns the kth ancestor of building v,
    # i.e. if we follow parent pointer k times from v.
    # In our chain, v itself is the "0th ancestor", then parent[v] is 1st, etc.
    def get_ancestor(v, k):
        cur = v
        bit = 0
        while k:
            if k & 1:
                cur = up[bit][cur]
                if cur == -1:
                    return -1
            k //= 2
            bit += 1
        return cur

    # Standard LCA in a forest (our forest will merge eventually into the highest building).
    # For two buildings u and v, they have a unique "visible chain" upward and they eventually merge.
    def lca(u, v):
        if u == -1 or v == -1:
            return -1
        if depth[u] < depth[v]:
            u, v = v, u
        diff = depth[u] - depth[v]
        bit = 0
        while diff:
            if diff & 1:
                u = up[bit][u]
            diff //= 2
            bit += 1
        if u == v:
            return u
        # lift both u and v so that their parent pointers become equal.
        for i in range(LOG-1, -1, -1):
            if up[i][u] != up[i][v]:
                u = up[i][u]
                v = up[i][v]
        return parent[u]

    # (4) Interpreting the problem in terms of these chains:
    # For each building i, the buildings visible from i to the east are exactly the nodes
    # in the chain: get_ancestor(i, 1), get_ancestor(i, 2), … .
    #
    # For a query (l, r) (remember l < r given as 1-indexed), we want the number of buildings j with j > r
    # that are visible from both buildings l and r.
    #
    # In our pointer‐tree, the set “visibleFrom(l)” equals the ancestor chain for l (i.e. follow parent pointer from l).
    # and similarly for r.
    # Their intersection is exactly the common ancestors of l and r. And because along the chain
    # indices strictly increase (f[i] > i always), every common ancestor will satisfy j > l,
    # and in fact if a common ancestor is not greater than r then it cannot be to the east of building r.
    #
    # Thus answer for a query (l, r) becomes:
    #   Let L = LCA(l, r)   (using our 0-indexed values for l and r)
    #   Count how many nodes in the chain for L (i.e. L, parent(L), parent(parent(L)), …) have index > r.
    #
    # We can “binary search” on the chain.
    # Note: the chain starting at node v, that is:
    #   A[0] = v, A[1] = parent[v], A[2] = parent(parent[v]), …, up to the root.
    # Because parent's index is always greater than its child’s index, the list A is strictly increasing.
    # So we want to find the smallest k such that A[k] > r.
    # Then answer = (total length of chain) - k.
    #
    # We'll implement this "binary search on the chain" using get_ancestor.
    def count_chain_above(v, r_idx):
        # total number of nodes in chain starting at v:
        total = depth[v] + 1  # because chain is: v at step=0, then depth[v] steps upward.
        lo = 0
        hi = total
        ans = total  # default if no node in the chain is <= r.
        while lo < hi:
            mid = (lo + hi) // 2
            a = get_ancestor(v, mid)
            if a == -1:
                lo = mid + 1
                continue
            if a > r_idx:
                ans = mid
                hi = mid
            else:
                lo = mid + 1
        # All chain nodes starting at index 'ans' have building index > r_idx.
        # If ans equals total it means none of the nodes in the chain is > r_idx.
        return total - ans if ans < total else 0

    # (5) Process each query.
    # Input queries are given 1-indexed; we convert to 0-indexed.
    out_lines = []
    for _ in range(Q):
        l_i = int(next(it)) - 1
        r_i = int(next(it)) - 1
        common = lca(l_i, r_i)
        if common == -1:
            out_lines.append("0")
        else:
            # Count in the chain (starting at common) the number of nodes with index > r_i.
            out_lines.append(str(count_chain_above(common, r_i)))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()