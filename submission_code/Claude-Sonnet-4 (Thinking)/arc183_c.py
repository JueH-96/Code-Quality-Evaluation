def solve():
    MOD = 998244353
    
    N, M = map(int, input().split())
    constraints = []
    for _ in range(M):
        L, R, X = map(int, input().split())
        constraints.append((L-1, R-1, X-1))  # Convert to 0-indexed
    
    def backtrack(perm, pos):
        if pos == N:
            # Check all constraints
            for L, R, X in constraints:
                max_val = max(perm[L:R+1])
                if perm[X] == max_val:
                    return 0
            return 1
        
        count = 0
        used = set(perm[:pos])
        
        for val in range(1, N+1):
            if val not in used:
                perm[pos] = val
                
                # Early pruning: check constraints that involve only assigned positions
                valid = True
                for L, R, X in constraints:
                    # Check if we can fully evaluate this constraint
                    if R < pos + 1:  # All positions in range are assigned
                        max_val = max(perm[L:R+1])
                        if perm[X] == max_val:
                            valid = False
                            break
                
                if valid:
                    count = (count + backtrack(perm, pos + 1)) % MOD
                
                perm[pos] = 0  # Reset for backtracking
        
        return count
    
    perm = [0] * N
    return backtrack(perm, 0)

print(solve())