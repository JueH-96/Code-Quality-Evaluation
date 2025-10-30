import sys
sys.setrecursionlimit(10000)
MOD = 998244353


def kosaraju_scc(n, to):
    """
    to : list, to[v] = destination of the single outgoing edge of v
    returns
        comp_id : component number of every vertex   (0 … comp_cnt-1)
        groups  : list with the list of vertices of every component
    """
    sys.setrecursionlimit(10000)

    g_rev = [[] for _ in range(n)]
    for v in range(n):
        g_rev[to[v]].append(v)

    order = []
    seen = [False] * n

    def dfs1(v):
        seen[v] = True
        u = to[v]
        if not seen[u]:
            dfs1(u)
        order.append(v)

    for v in range(n):
        if not seen[v]:
            dfs1(v)

    comp_id = [-1] * n
    groups = []

    def dfs2(v, cid):
        comp_id[v] = cid
        for u in g_rev[v]:
            if comp_id[u] == -1:
                dfs2(u, cid)

    for v in reversed(order):
        if comp_id[v] == -1:
            cid = len(groups)
            groups.append([])
            dfs2(v, cid)

    for v in range(n):
        groups[comp_id[v]].append(v)

    return comp_id, groups


def main() -> None:
    N, M = map(int, sys.stdin.readline().split())
    A = [int(x) - 1 for x in sys.stdin.readline().split()]  # 0-based

    comp_id, groups = kosaraju_scc(N, A)
    C = len(groups)                        # number of components

    # build children lists in the condensation graph (rooted forest)
    children = [[] for _ in range(C)]      # children of a component
    out_deg = [0] * C                      # to detect the roots (cycles)

    for v in range(N):
        p = comp_id[v]
        q = comp_id[A[v]]
        if p != q:
            children[q].append(p)
            out_deg[p] = 1                 # p has a parent

    # DP tables, lazily computed
    dp = [None] * C                        # dp[c][k] : 1 … M

    # prefix sums helper, returns list s[1 … M]
    def prefix(arr):
        ps = arr[:]
        acc = 0
        for i in range(1, M + 1):
            acc += ps[i]
            if acc >= MOD:
                acc -= MOD
            ps[i] = acc
        return ps

    def solve(c):
        if dp[c] is not None:
            return dp[c]

        if not children[c]:                # leaf
            res = [0] * (M + 1)
            for k in range(1, M + 1):
                res[k] = 1
            dp[c] = res
            return res

        # start with 1s, multiply child contributions
        res = [1] * (M + 1)
        for ch in children[c]:
            tbl = solve(ch)                # g[child][k]
            ps = prefix(tbl)               # Σ_{1 … k}
            for k in range(1, M + 1):
                res[k] = (res[k] * ps[k]) % MOD
        dp[c] = res
        return res

    answer = 1
    for c in range(C):
        if out_deg[c] == 0:                # root component (a cycle)
            tbl_root = solve(c)
            answer = (answer *
                       sum(tbl_root[1:])   # Σ_{k=1 … M} F(k)
                       ) % MOD

    print(answer % MOD)


if __name__ == "__main__":
    main()