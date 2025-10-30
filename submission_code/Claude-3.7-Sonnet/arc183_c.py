def solve():
    N, M = map(int, input().split())
    conditions = []
    for _ in range(M):
        L, R, X = map(int, input().split())
        conditions.append((L, R, X))
    
    MOD = 998244353
    
    # Precompute factorials and their inverses
    fact = [1]
    inv_fact = [1]
    for i in range(1, N + 1):
        fact.append((fact[-1] * i) % MOD)
        inv_fact.append(pow(fact[-1], MOD - 2, MOD))
    
    # Check if a subset of conditions can be violated simultaneously
    def can_violate_simultaneously(subset):
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                L_i, R_i, X_i = subset[i]
                L_j, R_j, X_j = subset[j]
                
                # Check if ranges overlap
                overlap_left = max(L_i, L_j)
                overlap_right = min(R_i, R_j)
                
                # If both X_i and X_j are in the overlap, we can't violate both conditions
                if overlap_left <= overlap_right and overlap_left <= X_i <= overlap_right and overlap_left <= X_j <= overlap_right:
                    return False
        return True
    
    # Calculate the number of permutations that violate all conditions in a subset
    def count_violating_permutations(subset):
        if not can_violate_simultaneously(subset):
            return 0
        
        # Create a directed graph to represent the constraints
        graph = [[] for _ in range(N + 1)]
        for L, R, X in subset:
            for pos in range(L, R + 1):
                if pos != X:
                    # Value at position X must be greater than value at position pos
                    graph[pos].append(X)
        
        # Use dynamic programming to count valid permutations
        # This is a simplified approach that works for the given constraints
        result = fact[N]
        for i in range(1, N + 1):
            indegree = sum(1 for j in range(1, N + 1) if i in graph[j])
            if indegree > 0:
                result = (result * inv_fact[indegree]) % MOD
        
        return result
    
    # Apply the principle of inclusion-exclusion
    answer = fact[N]
    for mask in range(1, 1 << M):
        subset = [conditions[i] for i in range(M) if (mask >> i) & 1]
        count = count_violating_permutations(subset)
        
        if bin(mask).count('1') % 2 == 1:
            answer = (answer - count) % MOD
        else:
            answer = (answer + count) % MOD
    
    return answer

print(solve())