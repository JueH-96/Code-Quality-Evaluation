import sys
def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    S = next(it)
    mod = 998244353

    # Number of real nodes = number of '(' = N//2
    M = S.count('(')
    # We'll use node IDs 1..M for real nodes, and 0 for the imaginary root.
    total_nodes = M + 1
    children = [[] for _ in range(total_nodes)]
    stack = [0]
    node_id = 0
    # Parse S into a rooted forest under imaginary root 0
    for ch in S:
        if ch == '(':
            node_id += 1
            parent = stack[-1]
            children[parent].append(node_id)
            stack.append(node_id)
        else:  # ch == ')'
            stack.pop()
    # Precompute factorials and inverse factorials up to N
    maxn = N
    fact = [1] * (maxn + 1)
    invfact = [1] * (maxn + 1)
    for i in range(1, maxn + 1):
        fact[i] = fact[i-1] * i % mod
    invfact[maxn] = pow(fact[maxn], mod-2, mod)
    for i in range(maxn, 0, -1):
        invfact[i-1] = invfact[i] * i % mod

    # Compute an "unordered‐shape" ID for each subtree by postorder.
    # Two subtrees get the same ID iff they are isomorphic as unordered rooted trees.
    shape_id = [0] * total_nodes
    # Map from sorted tuple of child‐shape‐IDs to a unique shape ID
    idmap = {(): 1}
    next_id = 2

    # Iterative postorder DFS
    stack2 = [(0, False)]
    while stack2:
        u, done = stack2.pop()
        if not done:
            # first time: push back marked, then push children
            stack2.append((u, True))
            for c in children[u]:
                stack2.append((c, False))
        else:
            # children of u have their shape_id set
            lst = [shape_id[c] for c in children[u]]
            lst.sort()
            key = tuple(lst)
            sid = idmap.get(key)
            if sid is None:
                sid = next_id
                idmap[key] = sid
                next_id += 1
            shape_id[u] = sid

    # Now count how many distinct ordered‐trees can be formed by permuting
    # children at each node, modulo identical‐shape children.
    ans = 1
    for u in range(total_nodes):
        c = len(children[u])
        if c <= 1:
            continue
        # number of ways = c! / (prod over each shape s of (mult_s)!)
        res = fact[c]
        freq = {}
        for v in children[u]:
            sid = shape_id[v]
            freq[sid] = freq.get(sid, 0) + 1
        for m in freq.values():
            if m > 1:
                res = res * invfact[m] % mod
        ans = ans * res % mod

    print(ans)

if __name__ == "__main__":
    main()