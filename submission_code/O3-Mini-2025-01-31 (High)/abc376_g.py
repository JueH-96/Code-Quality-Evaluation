def main():
    import sys, heapq
    sys.setrecursionlimit(10**6)
    mod = 998244353
    
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    t = int(data[0])
    index = 1
    out_lines = []
    
    # For each test case:
    for _ in range(t):
        n = int(data[index])
        index += 1
        # Build the tree.
        # Vertices are 0 ... n (vertex 0 is the root).
        children = [[] for _ in range(n+1)]
        # Read parents for vertices 1..n.
        # p_i is given for vertex i (1-indexed via input order) with 0 <= p_i < i.
        for i in range(1, n+1):
            p = int(data[index])
            index += 1
            children[p].append(i)
        # Read treasure weights a_i for vertices 1..n.
        a = [0] * (n+1)  # a[0] remains 0 since the treasure is never at the root.
        total_a = 0
        for i in range(1, n+1):
            val = int(data[index])
            index += 1
            a[i] = val
            total_a += val

        # Compute subtree sums.
        # For each vertex v, define subtree_sum[v] = a[v] + sum_{child in children[v]} subtree_sum[child].
        subtree_sum = [0] * (n+1)
        def dfs(v):
            s = a[v]  # For non-root vertices, a[v] is nonzero.
            for c in children[v]:
                s += dfs(c)
            subtree_sum[v] = s
            return s
        dfs(0)  # start DFS from root 0

        # The search process works as follows:
        # Initially only vertices whose parent is searched are "available". (Vertex 0 is searched at start.)
        # In an optimal strategy the next vertex to search is the “available” vertex whose subtree
        # (i.e. the branch that might contain the treasure) has the highest total a-value.
        #
        # Notice that any valid search order must be a linear extension of the partial order defined
        # by “parent comes before child”. The expected number of operations is the search position
        # (starting from 1) at which the treasure is found. In an optimal ordering the weighted sum
        # of positions ∑ (position(i) * a[i]) is minimized (over vertices 1..n).
        #
        # It turns out that the greedy strategy – always picking the available vertex with maximum
        # subtree_sum – yields the optimal ordering.
        
        # Simulate the greedy search order.
        # We use a max-heap (simulate with negatives) keyed by subtree_sum.
        heap = []
        for c in children[0]:
            heapq.heappush(heap, (-subtree_sum[c], c))
        
        # pos[i] will store the order (1-indexed) when vertex i is searched.
        pos = [0]*(n+1)
        order = 1
        while heap:
            neg_sum, v = heapq.heappop(heap)
            pos[v] = order
            order += 1
            for c in children[v]:
                heapq.heappush(heap, (-subtree_sum[c], c))
        
        # The expected number of operations is:
        #    E = (sum_{i=1}^n (pos[i] * a[i])) / (sum_{i=1}^n a[i])
        # We need to output the unique integer R (0 <= R < mod) such that R * (total_a) ≡ numerator mod mod.
        numerator = 0
        for i in range(1, n+1):
            numerator += pos[i] * a[i]
        inv_total = pow(total_a, mod-2, mod)
        ans = (numerator % mod) * inv_total % mod
        out_lines.append(str(ans))
    
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()