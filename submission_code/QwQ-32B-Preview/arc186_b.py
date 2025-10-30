def main():
    import sys
    MOD = 998244353
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Build children list
    children = [[] for _ in range(N+1)]
    for i in range(1, N+1):
        if A[i-1] > 0:
            children[A[i-1]].append(i)
    
    # DFS to compute subtree sizes
    subtree_size = [1] * (N+1)
    def dfs(node):
        size = 1
        for child in children[node]:
            if subtree_size[child] == 1:
                size += dfs(child)
            else:
                size += subtree_size[child]
        subtree_size[node] = size
        return size
    
    # Find roots
    roots = []
    for i in range(1, N+1):
        if A[i-1] == 0:
            if subtree_size[i] == 1:
                subtree_size[i] = dfs(i)
            roots.append(i)
    
    # Group subtree sizes
    from collections import defaultdict
    size_count = defaultdict(int)
    for root in roots:
        size_count[subtree_size[root]] += 1
    
    # Precompute factorial and inverse factorial
    fact = [1] * (N+1)
    for i in range(1, N+1):
        fact[i] = fact[i-1] * i % MOD
    inv_fact = [0] * (N+1)
    inv_fact[N] = pow(fact[N], MOD-2, MOD)
    for i in range(N, 0, -1):
        inv_fact[i-1] = inv_fact[i] * i % MOD
    
    # Compute the result
    result = 1
    for Si, Ki in size_count.items():
        inv_Si = pow(Si, MOD-2, MOD)
        term = fact[N] * inv_fact[N - Ki*Si] % MOD
        term = term * pow(inv_Si, Ki, MOD) % MOD
        term = term * inv_fact[Ki] % MOD
        result = result * term % MOD
    
    print(result)

if __name__ == "__main__":
    main()