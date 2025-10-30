import sys
from itertools import permutations

def cost_to_change_func(N, M_G, G_edges, M_H, H_edges, costs):
    """
    Calculate the minimum cost to make graphs G and H isomorphic.
    """
    min_cost = float('inf')
    
    # Helper function to build and calculate the cost for a specific permutation.
    def build_and_calculate_cost(permutation):
        perm_G_edges = set((permutation[i-1], permutation[j-1]) for i, j in G_edges)
        perm_H_edges = set((permutation[i-1], permutation[j-1]) for i, j in H_edges)
        cost = 0
        
        # Calculate the cost for edges that should exist but don't and vice versa.
        for i in range(N):
            for j in range(i+1, N):
                if not (i+1, j+1) in perm_G_edges and (i+1, j+1) in perm_H_edges:
                    cost += costs[i][j]
                elif (i+1, j+1) in perm_G_edges and not (i+1, j+1) in perm_H_edges:
                    cost += costs[i][j]
        return cost
    
    # Generate all permutations of N vertices.
    for perm in permutations(range(1, N+1)):
        current_cost = build_and_calculate_cost(perm)
        min_cost = min(min_cost, current_cost)
    
    return min_cost

# Read input from stdin
input = sys.stdin.readline
N = int(input())
M_G = int(input())
G_edges = [tuple(map(int, input().split())) for _ in range(M_G)]
M_H = int(input())
H_edges = [tuple(map(int, input().split())) for _ in range(M_H)]
costs = [list(map(int, input().split()))[i+1:] for i in range(N-1)]

# Calculate and print the output
print(cost_to_change_func(N, M_G, G_edges, M_H, H_edges, costs))