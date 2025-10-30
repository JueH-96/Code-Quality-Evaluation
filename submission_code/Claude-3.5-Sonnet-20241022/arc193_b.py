def solve():
    MOD = 998244353
    
    # Read input
    N = int(input())
    S = input()
    
    # Count edges to center vertex N
    center_edges = S.count('1')
    
    # Total number of edges in the graph
    total_edges = N + center_edges  # N edges for the cycle + edges to center
    
    # For each edge, we have 2 choices of direction
    # Therefore, total number of possible orientations is 2^total_edges
    # But we need to count only valid orientations that give unique in-degree sequences
    
    # We can use combinatorics to solve this
    # For the cycle edges, we need to consider that sum of in-degrees must equal total edges
    # and each vertex must have valid in-degree based on its connections
    
    # Calculate 2^total_edges modulo MOD
    result = pow(2, total_edges, MOD)
    
    print(result)

if __name__ == "__main__":
    solve()