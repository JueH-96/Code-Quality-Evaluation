# YOUR CODE HERE
import sys
import sys
import sys
def solve():
    import sys
    sys.setrecursionlimit(1 << 25)
    mod = 998244353
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx +=1
    results = []
    for _ in range(T):
        N = int(input[idx])
        idx +=1
        p = list(map(int, input[idx:idx+N]))
        idx +=N
        a = list(map(int, input[idx:idx+N]))
        idx +=N
        # Build tree
        tree = [[] for _ in range(N+1)]
        for i in range(1,N+1):
            parent = p[i-1]
            tree[parent].append(i)
        # Compute subtree sums
        subtree_sum = [0]*(N+1)
        # Assign a_i to nodes 1..N
        node_a = [0]*(N+1)
        for i in range(1,N+1):
            node_a[i] = a[i-1]
        def dfs_sum(u):
            s = node_a[u]
            for v in tree[u]:
                s += dfs_sum(v)
            subtree_sum[u] = s
            return s
        dfs_sum(0)
        # Sort children
        for u in range(N+1):
            tree[u].sort(key=lambda x: -subtree_sum[x])
        # Assign steps
        step_i = [0]*(N+1)
        step = 1
        def dfs_step(u):
            nonlocal step
            for v in tree[u]:
                step_i[v] = step
                step +=1
                dfs_step(v)
        dfs_step(0)
        # Compute sum a_i * step_i
        total = 0
        S = 0
        for i in range(1,N+1):
            total += a[i-1] * step_i[i]
            S +=a[i-1]
        total %= mod
        S %= mod
        inv_S = pow(S, mod-2, mod)
        R = total * inv_S % mod
        results.append(str(R))
    print('
'.join(results))
solve()