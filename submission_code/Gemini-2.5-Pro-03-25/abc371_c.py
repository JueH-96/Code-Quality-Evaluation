# YOUR CODE HERE
import sys
import itertools

# Read input functions for cleaner code
def read_int():
    """Reads a single integer from stdin."""
    return int(sys.stdin.readline())

def read_ints():
    """Reads multiple integers from a line in stdin separated by spaces."""
    return map(int, sys.stdin.readline().split())

def solve():
    """Solves the graph isomorphism minimum cost problem."""
    N = read_int()
    
    # Adjacency matrix for G (0-indexed)
    # Initialize an N x N matrix with False values.
    adjG = [[False]*N for _ in range(N)]
    MG = read_int()
    # Read edges of graph G and update the adjacency matrix.
    for _ in range(MG):
        u, v = read_ints()
        u -= 1 # Adjust from 1-based to 0-based indexing
        v -= 1
        # Store edge symmetrically since the graph is undirected.
        adjG[u][v] = True
        adjG[v][u] = True

    # Adjacency matrix for H (0-indexed)
    # Initialize an N x N matrix with False values.
    adjH = [[False]*N for _ in range(N)]
    MH = read_int()
    # Read edges of graph H and update the adjacency matrix.
    for _ in range(MH):
        a, b = read_ints()
        a -= 1 # Adjust from 1-based to 0-based indexing
        b -= 1
        adjH[a][b] = True
        adjH[b][a] = True

    # Costs dictionary mapping (u, v) pair with u < v (0-indexed) to cost A_{u+1, v+1}.
    costs = {}
    # Read all costs into a flat list first. The costs are given in multiple lines.
    current_costs_list = []
    # There are N-1 lines of costs to read according to the input format.
    for i in range(N - 1): 
         line_costs = list(read_ints())
         current_costs_list.extend(line_costs)

    # Populate the costs dictionary. The keys are pairs (i, j) with 0 <= i < j < N.
    # The pairs are generated in lexicographical order: (0,1), (0,2), ..., (0,N-1), (1,2), ..., (N-2, N-1).
    # This order matches the order in which costs are provided in the input.
    k = 0 # Index for current_costs_list
    for i in range(N):
        for j in range(i + 1, N):
             # Store cost for edge (i,j) using 0-indexed vertices i, j where i < j.
             costs[(i, j)] = current_costs_list[k]
             k += 1

    # Initialize minimum total cost to positive infinity. This will be updated as we find costs for permutations.
    min_total_cost = float('inf')

    # List of vertices {0, 1, ..., N-1} for generating permutations.
    vertices = list(range(N))
    
    # The constraints state N >= 1.
    # Handle the base case N=1 implicitly:
    # If N=1, vertices = [0]. itertools.permutations([0]) yields one permutation: ((0,)).
    # The inner loops `for i in range(N)` and `for j in range(i + 1, N)` do not execute any iterations because N=1.
    # Therefore, `current_total_cost` remains 0.
    # `min_total_cost` is updated to min(inf, 0) = 0. This is the correct answer for N=1.

    # Iterate through all possible permutations P of the vertex set {0, ..., N-1}.
    # Each permutation represents a potential mapping from vertices of G to vertices of H for isomorphism.
    for p in itertools.permutations(vertices):
        # p is a tuple representing the permutation. p[i] is the vertex in graph H 
        # that corresponds to vertex i in graph G under this potential isomorphism.
        
        current_total_cost = 0
        
        # Calculate the total cost required to modify graph H such that it becomes
        # isomorphic to graph G via the current permutation p.
        # Iterate through all unique pairs of vertices (i, j) with i < j in graph G.
        # There are N*(N-1)/2 such pairs.
        for i in range(N):
            for j in range(i + 1, N):
                # Check if an edge exists between vertices i and j in graph G.
                G_has_edge = adjG[i][j]
                
                # Find the corresponding vertices in graph H space based on permutation p:
                # Vertex i in G maps to vertex p[i] in H.
                # Vertex j in G maps to vertex p[j] in H.
                u, v = p[i], p[j]
                
                # Ensure u < v to use a canonical representation for the pair (edge).
                # This is necessary for consistent lookup in the adjacency matrix adjH and the costs dictionary.
                if u > v:
                    u, v = v, u 
                
                # Check if an edge exists between vertices u and v in the original graph H.
                H_has_edge = adjH[u][v]
                
                # The isomorphism condition requires that an edge (i, j) exists in G if and only if
                # an edge (p[i], p[j]) exists in the *modified* graph H (let's call it H').
                # If the current state of the edge (u, v) in the original H (H_has_edge) does not match 
                # the required state dictated by G's edge (i, j) (G_has_edge),
                # then an operation (add or remove edge) is needed on edge (u, v) in H to transform it into H'.
                # The cost for this operation is given by costs[(u, v)].
                if G_has_edge != H_has_edge:
                    current_total_cost += costs[(u, v)]
        
        # Update the overall minimum cost found across all permutations considered so far.
        min_total_cost = min(min_total_cost, current_total_cost)

    # After checking all N! permutations, min_total_cost holds the minimum cost required 
    # to make H isomorphic to G.
    # Print the result.
    # Since N >= 1, min_total_cost will always be assigned a finite non-negative value.
    print(min_total_cost)

# Execute the solver function when the script is run
solve()