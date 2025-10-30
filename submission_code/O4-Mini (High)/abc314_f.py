import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    MOD = 998244353

    N = int(input())
    p = [0] * (N - 1)
    q = [0] * (N - 1)
    for i in range(N - 1):
        pi, qi = map(int, input().split())
        p[i] = pi
        q[i] = qi

    # Precompute modular inverses up to N
    inv = [0] * (N + 1)
    inv[1] = 1
    for i in range(2, N + 1):
        # inv[i] = -(MOD//i) * inv[MOD%i] mod MOD
        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD

    # DSU for merging, plus comp_node to track the merge-tree node for each DSU rep
    parent = list(range(N + 1))
    size = [1] * (N + 1)
    comp_node = list(range(N + 1))  # comp_node[rep] = index in merge-tree

    def find(x):
        # Path compression
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Prepare merge-tree structures
    # Nodes are 1..N leaves, N+1..2N-1 internal
    max_node = 2 * N
    children = [[] for _ in range(max_node + 1)]
    # We'll create internal nodes in order N+1, N+2, ..., N+(N-1)=2N-1
    next_node = N + 1

    # Process merges
    for i in range(N - 1):
        u = p[i]
        v = q[i]
        ru = find(u)
        rv = find(v)
        # component-tree nodes
        cu = comp_node[ru]
        cv = comp_node[rv]
        # sizes
        a = size[ru]
        b = size[rv]
        s = a + b
        inv_s = inv[s]
        w_u = a * inv_s % MOD
        w_v = b * inv_s % MOD
        # new internal node
        new_node = next_node
        next_node += 1
        # add children with their weights
        children[new_node].append((cu, w_u))
        children[new_node].append((cv, w_v))
        # union in DSU, attach smaller to larger
        if size[ru] < size[rv]:
            ru, rv = rv, ru
        # now ru is new rep
        parent[rv] = ru
        size[ru] += size[rv]
        # the new component's merge-tree node is new_node
        comp_node[ru] = new_node

    # find root of merge-tree
    # any node 1..N is in final component
    root_rep = find(1)
    root_node = comp_node[root_rep]

    # dp[node] = sum of weights from root to this node
    dp = [0] * (max_node + 1)
    # iterative DFS
    stack = [root_node]
    dp[root_node] = 0
    while stack:
        cur = stack.pop()
        for (ch, w) in children[cur]:
            dp[ch] = dp[cur] + w
            if dp[ch] >= MOD:
                dp[ch] -= MOD
            stack.append(ch)

    # output for leaves 1..N
    out = sys.stdout
    res = []
    for i in range(1, N + 1):
        res.append(str(dp[i]))
    out.write(" ".join(res))

if __name__ == "__main__":
    main()