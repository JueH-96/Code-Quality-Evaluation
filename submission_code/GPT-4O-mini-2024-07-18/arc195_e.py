def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    # Read N and Q
    index = 0
    N = int(data[index])
    Q = int(data[index + 1])
    index += 2
    
    # Read A array
    A = list(map(int, data[index:index + N - 1]))
    index += N - 1
    
    # Read queries
    queries = []
    for _ in range(Q):
        u = int(data[index])
        v = int(data[index + 1])
        queries.append((u, v))
        index += 2
    
    # Precompute the total contribution of each edge A_i
    # Each edge A_i contributes to the distance between nodes in a specific way
    # For each edge A_i, it connects node i+1 to its parent P_i (which can be any of 1 to i-1)
    
    # The contribution of A_i to the distance between any two nodes u and v
    # is determined by how many times it is included in the paths between u and v.
    
    # The contribution of A_i is:
    # - It contributes to the distance between u and v if u or v is in the subtree rooted at i+1
    # - The number of valid parent configurations (P) that keep u and v in the same subtree
    #   is determined by the number of ways to choose parents for nodes 2 to i.
    
    # Precompute factorials and inverse factorials for combinatorial calculations
    from math import factorial
    from functools import lru_cache
    
    # Precompute factorials and inverse factorials
    fact = [1] * (N + 1)
    for i in range(2, N + 1):
        fact[i] = fact[i - 1] * i % MOD
    
    inv_fact = [1] * (N + 1)
    inv_fact[N] = pow(fact[N], MOD - 2, MOD)
    for i in range(N - 1, 0, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD
    
    # Function to calculate combinations C(n, k)
    def comb(n, k):
        if k < 0 or k > n:
            return 0
        return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
    
    # Precompute contributions for each edge
    contributions = [0] * (N + 1)
    
    for i in range(2, N + 1):
        # A[i-1] corresponds to A_i
        contributions[i] = A[i - 2] * comb(i - 2, i - 2) % MOD
    
    # Prepare answers for each query
    results = []
    
    for u, v in queries:
        total_distance = 0
        
        # Calculate the contribution for each edge from u to v
        for i in range(2, N + 1):
            if u <= i <= v:
                total_distance = (total_distance + contributions[i]) % MOD
        
        results.append(total_distance)
    
    # Print results
    sys.stdout.write('
'.join(map(str, results)) + '
')