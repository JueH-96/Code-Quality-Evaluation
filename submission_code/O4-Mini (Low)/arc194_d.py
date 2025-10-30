import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10000)
    MOD = 998244353

    data = sys.stdin.read().split()
    N = int(data[0])
    S = data[1].strip()

    # Precompute factorials and inverse factorials up to N
    fact = [1]*(N+1)
    for i in range(1, N+1):
        fact[i] = fact[i-1]*i % MOD
    invfact = [1]*(N+1)
    invfact[N] = pow(fact[N], MOD-2, MOD)
    for i in range(N, 0, -1):
        invfact[i-1] = invfact[i]*i % MOD

    # Build the tree: we use a super-root = node 0
    # Each time we see '(', create a new node and push it; on ')', pop it.
    # The popped node becomes a child of the node now on top.
    children = [[] for _ in range(N+1)]
    stack = [0]  # super-root

    node_id = 0
    for ch in S:
        if ch == '(':
            node_id += 1
            u = node_id
            children[stack[-1]].append(u)
            stack.append(u)
        else:  # ch == ')'
            stack.pop()

    # We'll do a post-order DFS to compute dp[u] and a canonical shape-id[u].
    dp = [0]*(N+1)
    shape_id = [0]*(N+1)
    canonical_map = {}  # map tuple(child_shape_ids) -> unique small int
    next_shape_int = 1

    def dfs(u):
        nonlocal next_shape_int
        # First process all children
        for v in children[u]:
            dfs(v)
        # Gather children's shape_ids, sort them
        lst = sorted(shape_id[v] for v in children[u])
        # Make a tuple to identify the isomorphism class
        tpl = tuple(lst)
        if tpl not in canonical_map:
            canonical_map[tpl] = next_shape_int
            next_shape_int += 1
        shape_id[u] = canonical_map[tpl]

        # Compute dp[u]
        deg = len(children[u])
        res = fact[deg]  # deg(u)!
        # multiply in each child's dp
        for v in children[u]:
            res = (res * dp[v]) % MOD
        # divide by factorials of multiplicities of identical shape-ids
        # i.e. for each run of equal IDs in lst, divide by (run-length)!
        i = 0
        while i < deg:
            j = i+1
            while j < deg and lst[j] == lst[i]:
                j += 1
            cnt = j - i
            res = (res * invfact[cnt]) % MOD
            i = j

        dp[u] = res

    dfs(0)
    # The answer is dp at the super-root
    print(dp[0] % MOD)

if __name__ == "__main__":
    main()