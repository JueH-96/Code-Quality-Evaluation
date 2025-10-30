from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    
    # Build adjacency list
    adj = defaultdict(list)
    for _ in range(M):
        u, v = map(int, input().split())
        u -= 1
        v -= 1
        adj[u].append(v)
        adj[v].append(u)
    
    W = list(map(int, input().split()))
    A = list(map(int, input().split()))
    
    def get_valid_subsets(vertex):
        # Get all valid subsets of neighbors where sum of weights < W[vertex]
        neighbors = adj[vertex]
        valid = []
        
        # Try all possible subsets using bit manipulation
        for mask in range(1 << len(neighbors)):
            subset = []
            weight_sum = 0
            for i in range(len(neighbors)):
                if mask & (1 << i):
                    neighbor = neighbors[i]
                    subset.append(neighbor)
                    weight_sum += W[neighbor]
            
            if weight_sum < W[vertex]:
                valid.append(subset)
                
        return valid
    
    # For each vertex, precompute all valid subsets of neighbors
    valid_moves = []
    for i in range(N):
        valid_moves.append(get_valid_subsets(i))
    
    def get_max_moves(pieces):
        if sum(pieces) == 0:
            return 0
            
        result = 0
        
        # Try removing piece from each vertex
        for vertex in range(N):
            if pieces[vertex] == 0:
                continue
                
            # Try each valid subset of neighbors
            for subset in valid_moves[vertex]:
                # Make move
                new_pieces = pieces[:]
                new_pieces[vertex] -= 1
                for neighbor in subset:
                    new_pieces[neighbor] += 1
                    
                # Recursively try remaining moves
                result = max(result, 1 + get_max_moves(new_pieces))
                
        return result
    
    print(get_max_moves(A))

solve()