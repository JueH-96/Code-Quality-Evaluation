import sys
from itertools import combinations

# DSU data structure
class DSU:
    def __init__(self, n_vertices):
        # Vertices are 0 to n_vertices-1
        self.parent = list(range(n_vertices))
        self.num_components = n_vertices
        # For N <= 8, union by rank/size is not strictly necessary for performance.
        # Path compression alone is generally good enough for small N.
        # self.rank = [0] * n_vertices # if using union by rank

    def find(self, i):
        if self.parent[i] == i:
            return i
        # Path compression
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # Simple union: make root_i child of root_j (or vice versa).
            self.parent[root_i] = root_j 
            self.num_components -= 1
            return True # Merge occurred
        return False # Already in same set

def solve():
    N, M, K = map(int, sys.stdin.readline().split())
    
    all_edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        all_edges.append((u, v, w)) # Store edges as (u, v, w)

    min_cost_mod_K = K 
    # Initialize with K. Any valid cost is in [0, K-1].
    # If K=1, min_cost_mod_K will be initialized to 1. The actual cost will be sum % 1 = 0.
    # So min_cost_mod_K correctly becomes 0.

    # A spanning tree for N vertices has N-1 edges.
    num_edges_to_select = N - 1

    # Iterate over all combinations of (N-1) edges chosen from the M available edges.
    for current_edge_selection_tuple in combinations(all_edges, num_edges_to_select):
        current_sum_weights = 0
        for edge in current_edge_selection_tuple:
            # edge is a tuple (u, v, w)
            current_sum_weights += edge[2] # Sum the weight w

        # Check if this selection of N-1 edges forms a spanning tree.
        # A DSU structure is used for this check.
        # Vertices are 1 to N in problem, DSU uses 0 to N-1.
        dsu = DSU(N) 
        
        for edge in current_edge_selection_tuple:
            u_original, v_original, _ = edge # Unpack edge details
            
            # Adjust vertex numbers from 1-indexed (problem) to 0-indexed (DSU)
            u_idx = u_original - 1
            v_idx = v_original - 1
            
            # Perform union. The DSU's internal num_components count is updated.
            dsu.union(u_idx, v_idx)
        
        # A set of N-1 edges on N vertices forms a spanning tree if and only if
        # it connects all N vertices. This means after processing all N-1 edges,
        # the DSU should report exactly 1 connected component.
        if dsu.num_components == 1:
            cost = current_sum_weights % K
            if cost < min_cost_mod_K:
                min_cost_mod_K = cost
            
            # Optimization: if cost becomes 0, it's the absolute minimum possible.
            if min_cost_mod_K == 0:
                break
    
    print(min_cost_mod_K)

# Ensure the script runs the solve function when executed
if __name__ == '__main__':
    solve()