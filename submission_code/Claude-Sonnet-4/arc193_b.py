MOD = 998244353

def solve():
    N = int(input())
    s = input().strip()
    
    # Count star edges (connections to vertex N)
    k = sum(int(c) for c in s)
    
    # Key insight: we need to count distinct in-degree sequences
    # Let's use the fact that the sum of in-degrees equals total edges = N + k
    
    # For small cases, we can use DP or direct enumeration
    # For larger cases, we need to find patterns
    
    if N <= 20:  # Small case - direct enumeration
        result = set()
        total_edges = N + k
        star_positions = [i for i in range(N) if s[i] == '1']
        
        for mask in range(1 << total_edges):
            in_degrees = [0] * (N + 1)
            edge_idx = 0
            
            # Cycle edges
            for i in range(N):
                next_i = (i + 1) % N
                if mask & (1 << edge_idx):
                    in_degrees[next_i] += 1
                else:
                    in_degrees[i] += 1
                edge_idx += 1
            
            # Star edges
            for pos in star_positions:
                if mask & (1 << edge_idx):
                    in_degrees[N] += 1
                else:
                    in_degrees[pos] += 1
                edge_idx += 1
            
            result.add(tuple(in_degrees))
        
        return len(result) % MOD
    
    # For larger N, we need a mathematical approach
    # The answer depends on the structure of the graph
    
    # After analysis, the formula involves:
    # - Ways to orient the cycle (affects distribution among vertices 0..N-1)
    # - Ways to orient star edges (affects vertex N and star vertices)
    
    # This is a complex combinatorial problem
    # Let me implement a more efficient solution
    
    # Use generating functions or DP on smaller state space
    from collections import defaultdict
    
    # DP state: possible in-degree distributions
    dp = defaultdict(int)
    dp[tuple([0] * (N + 1))] = 1
    
    # Process cycle edges
    for i in range(N):
        new_dp = defaultdict(int)
        next_i = (i + 1) % N
        
        for state, count in dp.items():
            state_list = list(state)
            
            # Option 1: edge goes i -> next_i
            state_list[next_i] += 1
            new_dp[tuple(state_list)] = (new_dp[tuple(state_list)] + count) % MOD
            state_list[next_i] -= 1
            
            # Option 2: edge goes next_i -> i
            state_list[i] += 1
            new_dp[tuple(state_list)] = (new_dp[tuple(state_list)] + count) % MOD
        
        dp = new_dp
    
    # Process star edges
    for i in range(N):
        if s[i] == '1':
            new_dp = defaultdict(int)
            
            for state, count in dp.items():
                state_list = list(state)
                
                # Option 1: edge goes i -> N
                state_list[N] += 1
                new_dp[tuple(state_list)] = (new_dp[tuple(state_list)] + count) % MOD
                state_list[N] -= 1
                
                # Option 2: edge goes N -> i
                state_list[i] += 1
                new_dp[tuple(state_list)] = (new_dp[tuple(state_list)] + count) % MOD
            
            dp = new_dp
    
    return len(dp) % MOD

print(solve())