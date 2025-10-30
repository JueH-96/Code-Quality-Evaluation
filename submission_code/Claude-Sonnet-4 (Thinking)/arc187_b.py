def solve():
    N, M = map(int, input().split())
    B = list(map(int, input().split()))
    
    MOD = 998244353
    
    # Find positions with -1
    minus_ones = [i for i in range(N) if B[i] == -1]
    q = len(minus_ones)
    
    def count_components(values):
        # Use Union-Find to count connected components
        parent = list(range(N))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
        
        # Add edges where values[i] <= values[j] for i < j
        for i in range(N):
            for j in range(i + 1, N):
                if values[i] <= values[j]:
                    union(i, j)
        
        # Count unique components
        return len(set(find(i) for i in range(N)))
    
    if q == 0:
        return count_components(B)
    
    # Use memoization with recursive approach
    memo = {}
    
    def dp(index, values_tuple):
        if index == q:
            values = list(values_tuple)
            return count_components(values)
        
        if (index, values_tuple) in memo:
            return memo[(index, values_tuple)]
        
        result = 0
        values = list(values_tuple)
        
        for val in range(1, M + 1):
            values[minus_ones[index]] = val
            result = (result + dp(index + 1, tuple(values))) % MOD
        
        memo[(index, values_tuple)] = result
        return result
    
    return dp(0, tuple(B))

print(solve())