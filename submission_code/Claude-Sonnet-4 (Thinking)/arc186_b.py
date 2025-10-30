def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    MOD = 998244353
    
    # Build edges for the DAG
    edges = []
    for i in range(1, N + 1):
        # Constraint 1: P_j > P_i for j with A_i < j < i
        for j in range(A[i - 1] + 1, i):
            edges.append((i, j))
        # Constraint 2: P_{A_i} < P_i if A_i > 0
        if A[i - 1] > 0:
            edges.append((A[i - 1], i))
    
    # Use memoization to count linear extensions
    memo = {}
    
    def count_linear_extensions(vertices_mask):
        if vertices_mask in memo:
            return memo[vertices_mask]
        
        # Get list of remaining vertices
        vertices = [i for i in range(1, N + 1) if vertices_mask & (1 << i)]
        
        if len(vertices) <= 1:
            return 1
        
        # Find minimal elements (vertices with no incoming edges from remaining vertices)
        minimal = []
        for v in vertices:
            is_minimal = True
            for u, w in edges:
                if w == v and (vertices_mask & (1 << u)):
                    is_minimal = False
                    break
            if is_minimal:
                minimal.append(v)
        
        result = 0
        for v in minimal:
            new_mask = vertices_mask ^ (1 << v)
            result = (result + count_linear_extensions(new_mask)) % MOD
        
        memo[vertices_mask] = result
        return result
    
    # Start with all vertices from 1 to N
    full_mask = (1 << (N + 1)) - 2
    return count_linear_extensions(full_mask)

print(solve())