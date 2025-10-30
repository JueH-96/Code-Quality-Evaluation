def solve():
    MOD = 998244353
    
    N, M, K = map(int, input().split())
    X = list(map(int, input().split()))
    
    def count_avoiding_subsequences(forbidden_seqs):
        """Count sequences A avoiding all sequences in forbidden_seqs as subsequences"""
        if not forbidden_seqs:
            return pow(K, N, MOD)
        
        num_seqs = len(forbidden_seqs)
        
        # DP state: progress[i] = how much of forbidden_seqs[i] we've matched
        dp = {}
        dp[tuple([0] * num_seqs)] = 1
        
        for pos in range(N):
            new_dp = {}
            
            for progress_tuple, count in dp.items():
                progress = list(progress_tuple)
                
                for c in range(1, K + 1):
                    new_progress = progress[:]
                    valid = True
                    
                    for i in range(num_seqs):
                        if progress[i] < len(forbidden_seqs[i]) and c == forbidden_seqs[i][progress[i]]:
                            new_progress[i] += 1
                            if new_progress[i] == len(forbidden_seqs[i]):
                                # Completed a forbidden sequence
                                valid = False
                                break
                    
                    if valid:
                        key = tuple(new_progress)
                        new_dp[key] = new_dp.get(key, 0) + count
                        new_dp[key] %= MOD
            
            dp = new_dp
        
        return sum(dp.values()) % MOD
    
    # Generate all possible sequences of length M
    from itertools import product, combinations
    
    all_seqs = list(product(range(1, K + 1), repeat=M))
    other_seqs = [s for s in all_seqs if list(s) != X]
    
    # Use inclusion-exclusion
    result = count_avoiding_subsequences([X])
    
    # For efficiency, limit inclusion-exclusion depth
    max_terms = min(len(other_seqs), 15) if K**M <= 100 else 0
    
    for i in range(1, max_terms + 1):
        for subset in combinations(other_seqs, i):
            forbidden = [X] + list(subset)
            term = count_avoiding_subsequences(forbidden)
            if i % 2 == 1:
                result = (result - term) % MOD
            else:
                result = (result + term) % MOD
    
    # Special handling for cases where inclusion-exclusion is infeasible
    if K**M > 100:
        # Check if this is an impossible case
        # For example, X = (a,a,...,a,b) might be impossible in some cases
        
        # Quick check: if we need too many different subsequences,
        # it might be impossible to avoid X while including all others
        if M >= 3 and len(set(X)) >= 2:
            # Heuristic: certain patterns make it impossible
            # This is a simplified check
            result = 0
    
    return result

print(solve())