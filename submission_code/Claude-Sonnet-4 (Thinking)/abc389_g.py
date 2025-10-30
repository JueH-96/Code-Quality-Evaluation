def solve():
    N, P = map(int, input().split())
    
    from itertools import combinations
    from math import comb
    
    def count_connected_bipartite(n_A, n_B, M):
        max_edges = n_A * n_B
        
        if M > max_edges:
            return 0
        
        total = comb(max_edges, M)
        
        # Apply inclusion-exclusion to count connected graphs
        # Subtract graphs where some vertices are isolated
        for mask in range(1, 2**(n_A + n_B)):
            isolated_A = 0
            isolated_B = 0
            
            for i in range(n_A):
                if mask & (1 << i):
                    isolated_A += 1
            
            for i in range(n_B):
                if mask & (1 << (n_A + i)):
                    isolated_B += 1
            
            # Calculate remaining edges after removing isolated vertices
            remaining_edges = (n_A - isolated_A) * (n_B - isolated_B)
            
            if M <= remaining_edges:
                contribution = comb(remaining_edges, M)
                if (isolated_A + isolated_B) % 2 == 1:
                    total = (total - contribution) % P
                else:
                    total = (total + contribution) % P
        
        return total % P
    
    results = []
    
    for M in range(N-1, N*(N-1)//2 + 1):
        total_count = 0
        
        # Number of ways to choose N/2 - 1 vertices from {2, 3, ..., N} 
        # to be in part A with vertex 1
        num_partitions = comb(N-1, N//2 - 1)
        
        # For each partition, count connected bipartite graphs
        count_per_partition = count_connected_bipartite(N//2, N//2, M)
        
        total_count = (num_partitions * count_per_partition) % P
        results.append(total_count)
    
    print(' '.join(map(str, results)))

solve()