# YOUR CODE HERE
import sys

def main():
    """
    This function contains the main logic for solving the problem.
    It reads the graph specification, builds a Minimum Spanning Tree using a
    Kruskal-like approach on the implicit graph, and prints the result.
    """

    # Set a higher recursion limit for the recursive DSU implementation.
    # The default limit might be too low for N up to 2 * 10^5.
    sys.setrecursionlimit(2 * 10**5 + 5)
    
    # Use sys.stdin.readline for faster I/O in competitive programming.
    input = sys.stdin.readline

    class DSU:
        """
        A Disjoint Set Union (DSU) data structure, also known as Union-Find.
        This implementation uses path compression and union by size for optimization.
        It's designed for 1-based indexing of vertices (from 1 to N).
        """
        def __init__(self, n):
            # parent[i] stores the parent of element i. A root has parent[i] == i.
            self.parent = list(range(n + 1))
            # size[i] stores the size of the set rooted at i. Used for union by size.
            self.size = [1] * (n + 1)
            
        def find(self, i):
            """
            Finds the representative (root) of the set containing element i.
            Uses path compression to flatten the structure for future finds.
            """
            if self.parent[i] == i:
                return i
            # Path compression: set the parent of i directly to the root.
            self.parent[i] = self.find(self.parent[i])
            return self.parent[i]

        def union(self, i, j):
            """
            Merges the sets containing elements i and j.
            Uses union by size to keep the trees relatively balanced.
            Returns True if a merge occurred (i and j were in different sets),
            and False otherwise.
            """
            root_i = self.find(i)
            root_j = self.find(j)
            
            if root_i != root_j:
                # Union by size: attach the smaller tree to the root of the larger tree.
                if self.size[root_i] < self.size[root_j]:
                    root_i, root_j = root_j, root_i
                
                self.parent[root_j] = root_i
                self.size[root_i] += self.size[root_j]
                return True
            
            return False

    # Read graph size N and number of operations M.
    try:
        line = input().strip()
        if not line: return # Handles empty input at the end of file
        N, M = map(int, line.split())
    except (IOError, ValueError):
        return

    # Store operations to be sorted by cost. Each element is a tuple (cost, vertex_list).
    operations = []
    for _ in range(M):
        # Read K_i (size of vertex set) and C_i (cost).
        K, C = map(int, input().split())
        # Read the list of vertices A_i.
        A = list(map(int, input().split()))
        operations.append((C, A))

    # Sort operations by cost in ascending order. This is the core of simulating
    # Kruskal's algorithm without explicitly generating all edges.
    operations.sort()

    # Initialize DSU for N vertices.
    dsu = DSU(N)
    
    # total_weight will accumulate the cost of the MST.
    # edges_count tracks the number of edges added to the MST.
    total_weight = 0
    edges_count = 0

    # Process operations in order of increasing cost.
    for cost, vertices in operations:
        # A clique operation only forms edges if it involves at least two vertices.
        if len(vertices) > 1:
            # We connect all vertices in the set to the component of the first vertex.
            # This effectively connects all components represented in the vertex set.
            v_ref = vertices[0]
            for i in range(1, len(vertices)):
                v_curr = vertices[i]
                
                # If v_curr and v_ref are in different components, union them.
                # dsu.union returns True if a merge happened.
                if dsu.union(v_ref, v_curr):
                    # This merge corresponds to adding an edge of `cost` to the MST.
                    total_weight += cost
                    edges_count += 1
    
    # A graph with N vertices is connected if and only if its spanning forest
    # becomes a single spanning tree, which must contain exactly N - 1 edges.
    if edges_count == N - 1:
        print(total_weight)
    else:
        # If fewer than N - 1 edges were added, the graph is not connected.
        print(-1)

if __name__ == "__main__":
    main()