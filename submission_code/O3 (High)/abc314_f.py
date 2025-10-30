import sys

MOD = 998244353

# ----------  DSU that creates a new root on every union  ----------
def build_merge_tree(n, matches):
    """
    returns
        tot            : id of last node  (= root of the whole tree)
        size, lch, rch : arrays describing the merge tree
    leaves are 1..n, internal nodes are n+1 .. tot
    """
    max_nodes = 2 * n + 5
    size = [0] * max_nodes          # number of players in the sub-team
    lch  = [0] * max_nodes          # left  child
    rch  = [0] * max_nodes          # right child
    dsu  = [0] * max_nodes          # DSU parent used while building

    for i in range(1, n + 1):
        size[i] = 1
        dsu[i]  = i

    # path-compression find
    def find(x):
        while dsu[x] != x:
            dsu[x] = dsu[dsu[x]]
            x = dsu[x]
        return x

    tot = n
    for p, q in matches:
        a = find(p)
        b = find(q)
        tot += 1
        new = tot
        lch[new]  = a
        rch[new]  = b
        size[new] = size[a] + size[b]

        # make `new` the DSU root of the merged component
        dsu[a] = dsu[b] = new
        dsu[new] = new

    return tot, size, lch, rch


# ----------  main logic  ----------
def main() -> None:
    sys.setrecursionlimit(1_000_000)
    read = sys.stdin.readline

    n = int(read())
    matches = [tuple(map(int, read().split())) for _ in range(n - 1)]

    root, size, lch, rch = build_merge_tree(n, matches)

    # pre-compute modular inverses of 1 .. n
    inv = [0] * (n + 2)
    inv[1] = 1
    for i in range(2, n + 1):
        inv[i] = MOD - (MOD // i) * inv[MOD % i] % MOD

    dp = [0] * (2 * n + 5)      # expected wins accumulated so far
    stack = [root]              # iterative DFS, dp[root] = 0 by definition

    while stack:
        u = stack.pop()
        left = lch[u]
        if left == 0:           # leaf ⇒ nothing to propagate
            continue
        right = rch[u]

        inv_parent = inv[size[u]]

        dp[left]  = (dp[u] + size[left]  * inv_parent) % MOD
        dp[right] = (dp[u] + size[right] * inv_parent) % MOD

        stack.append(left)
        stack.append(right)

    # output answers for players 1 .. n (they correspond to leaf nodes 1 .. n)
    print(' '.join(str(dp[i]) for i in range(1, n + 1)))


if __name__ == '__main__':
    main()