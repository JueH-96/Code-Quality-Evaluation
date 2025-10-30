import sys
import itertools

class DSU:
    """
    Disjoint Set Union (DSU) data structure with path compression.
    Used to efficiently manage sets of elements and perform union/find operations.
    """
    def __init__(self, n):
        """
        Initializes the DSU structure for n elements.
        Each element is initially in its own set.
        """
        self.parent = list(range(n))
        self.num_components = n # Tracks the number of disjoint sets

    def find(self, i):
        """
        Finds the representative (root) of the set containing element i.
        Performs path compression to flatten the tree structure for faster future lookups.
        """
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """
        Merges the sets containing elements i and j.
        Returns True if the sets were different and merged, False otherwise (i.e., i and j were already in the same set).
        If merging occurs, decrements the count of components.
        """
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Attach the root of one tree to the root of the other.
            # (Union by rank/size could be added for slightly better theoretical performance,
            # but for N <= 8, simple union with path compression is perfectly adequate.)
            self.parent[root_i] = root_j
            self.num_components -= 1 # One less component after merging
            return True # Successfully merged
        return False # i and j were already in the same set (adding this edge would create a cycle)

def solve():
    """
    Reads the graph input and finds the minimum cost of a spanning tree
    where cost is defined as the sum of edge weights modulo K.
    """
    # Read N (number of vertices), M (number of edges), K (modulo value)
    N, M, K = map(int, sys.stdin.readline().split())

    edges = []
    for _ in range(M):
        u, v, w = map(int, sys.stdin.readline().split())
        # Store edges with 0-indexed vertices for easier array access in Python.
        # Vertices are 1-indexed in input (1 to N).
        edges.append((u - 1, v - 1, w))

    # Initialize min_cost. Any cost modulo K will be in [0, K-1].
    # K itself is a safe initial upper bound, as 0 <= w_i < K ensures sum % K < K.
    min_cost = K 

    # A spanning tree for a graph with N vertices must have exactly N-1 edges.
    # We iterate through all combinations of N-1 edges from the M available edges.
    # itertools.combinations generates combinations efficiently.
    for selected_edges_tuple in itertools.combinations(edges, N - 1):
        dsu = DSU(N) # Create a new DSU instance for each combination
        current_sum_weight = 0
        
        # Sum weights and try to connect vertices using the selected edges
        for u, v, w in selected_edges_tuple:
            current_sum_weight += w
            # Attempt to unite the sets containing u and v.
            # If u and v are already in the same set (union returns False),
            # it means adding this edge would create a cycle.
            # For a set of N-1 edges, if a cycle forms, it means the graph will not
            # connect all N vertices with these N-1 edges, thus it's not a spanning tree.
            # The dsu.num_components check at the end correctly identifies this case.
            dsu.union(u, v) 
            
        # After processing all N-1 edges in the current combination:
        # A set of N-1 edges forms a spanning tree if and only if it connects all N vertices.
        # If dsu.num_components is 1, it means all vertices are connected.
        # Since it's connected and has N-1 edges, it must also be acyclic (thus, a tree).
        if dsu.num_components == 1:
            current_cost = current_sum_weight % K
            min_cost = min(min_cost, current_cost)
    
    # Print the minimum cost found among all valid spanning trees.
    print(min_cost)

# Call the solve function to run the program
solve()