def solve():
    MOD = 998244353
    
    n = int(input())
    A = list(map(int, input().split()))
    
    # For each position i, we need to satisfy:
    # 1. P_j > P_i for A_i < j < i
    # 2. P_{A_i} < P_i if A_i > 0
    
    # We'll use DP with memoization
    # State: (position, mask of used values)
    # But this is too expensive for n=3*10^5
    
    # Alternative approach: think about the structure
    # The constraints create a partial order on positions
    # We need to count linear extensions of this partial order
    
    # Let's build a graph of constraints and use inclusion-exclusion
    # or find a more direct combinatorial solution
    
    # Actually, let's think differently:
    # For each position i, we know some constraints about P_i
    # We can process positions and count valid assignments
    
    from functools import lru_cache
    
    # Build constraint graph
    constraints = []  # (i, j) means P_i < P_j
    
    for i in range(n):
        pos = i + 1  # 1-indexed position
        if A[i] > 0:
            constraints.append((A[i], pos))  # P_{A_i} < P_i
        
        # For j where A_i < j < i, we need P_j > P_i
        for j in range(A[i] + 1, pos):
            constraints.append((pos, j))  # P_i < P_j
    
    # Now we need to count topological sorts of this DAG
    # But this is still complex for large n
    
    # Let me try a different approach based on the structure
    # Process positions in order and count valid values at each step
    
    @lru_cache(maxsize=None)
    def dp(pos, used_mask):
        if pos > n:
            return 1
        
        result = 0
        for val in range(1, n + 1):
            if used_mask & (1 << (val - 1)):
                continue
            
            # Check if this value satisfies constraints
            valid = True
            
            # Check constraint with A[pos-1]
            if A[pos - 1] > 0:
                # We need P_{A[pos-1]} < val
                # This will be checked when we assign P_{A[pos-1]}
                pass
            
            # Check constraints with positions between A[pos-1] and pos-1
            # These positions should have values > val
            # This will be enforced when we assign those positions
            
            if valid:
                new_mask = used_mask | (1 << (val - 1))
                result = (result + dp(pos + 1, new_mask)) % MOD
        
        return result
    
    # This approach is still too slow for large n
    # Let me implement a more efficient solution
    
    # Actually, let's solve this step by step with a different DP
    memo = {}
    
    def count_ways(pos, available):
        if pos > n:
            return 1
        
        key = (pos, tuple(sorted(available)))
        if key in memo:
            return memo[key]
        
        result = 0
        
        for val in list(available):
            # Check if we can place val at position pos
            valid = True
            
            # Remove val from available
            new_available = available - {val}
            
            if valid:
                result = (result + count_ways(pos + 1, new_available)) % MOD
        
        memo[key] = result
        return result
    
    # For large n, we need a more efficient approach
    # Let's implement a direct combinatorial solution
    
    # Simpler approach: use the fact that constraints form a specific structure
    result = 1
    used = [False] * (n + 1)
    assignment = [0] * (n + 1)
    
    # Process positions in a specific order to respect constraints
    def backtrack(pos):
        if pos > n:
            return 1
        
        count = 0
        for val in range(1, n + 1):
            if used[val]:
                continue
            
            # Check if this assignment is valid
            valid = True
            
            # Check constraint: P_{A[pos-1]} < P_pos if A[pos-1] > 0
            if A[pos - 1] > 0 and assignment[A[pos - 1]] != 0:
                if assignment[A[pos - 1]] >= val:
                    valid = False
            
            # Check constraint: P_j > P_pos for A[pos-1] < j < pos
            for j in range(A[pos - 1] + 1, pos):
                if assignment[j] != 0 and assignment[j] <= val:
                    valid = False
            
            if valid:
                used[val] = True
                assignment[pos] = val
                count = (count + backtrack(pos + 1)) % MOD
                used[val] = False
                assignment[pos] = 0
        
        return count
    
    print(backtrack(1))

solve()