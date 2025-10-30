import sys

# It's good practice to increase the recursion limit for deep DSU paths,
# although path compression keeps the hierarchy very flat.
sys.setrecursionlimit(2 * 10**5 + 5)

class DSU:
    """
    Disjoint Set Union (DSU) data structure, also known as Union-Find.
    
    This implementation uses path compression and union by size for efficiency.
    It is adapted to maintain counts of nodes and edges within each disjoint set.
    """
    def __init__(self, n):
        # parent[i]: parent of element i.
        self.parent = list(range(n + 1))
        # num_nodes[i]: number of nodes in the component if i is the root.
        self.num_nodes = [1] * (n + 1)
        # num_edges[i]: number of edges within the component if i is the root.
        self.num_edges = [0] * (n + 1)

    def find(self, i):
        """Finds the representative (root) of the set containing element i."""
        if self.parent[i] == i:
            return i
        # Path compression
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Merges the sets containing elements i and j."""
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # Union by size: merge smaller component into larger one.
            if self.num_nodes[root_i] < self.num_nodes[root_j]:
                root_i, root_j = root_j, root_i
            
            self.parent[root_j] = root_i
            # Update counts for the new merged component.
            self.num_nodes[root_i] += self.num_nodes[root_j]
            self.num_edges[root_i] += self.num_edges[root_j] + 1
        else:
            # The edge is internal to an existing component.
            self.num_edges[root_i] += 1

def main():
    """
    Reads input, solves the problem, and prints the output.
    """
    input = sys.stdin.readline
    
    try:
        line = input().strip()
        if not line: return
        N, M = map(int, line.split())
    except (IOError, ValueError):
        return
        
    dsu = DSU(N)
    
    for _ in range(M):
        A, B = map(int, input().split())
        dsu.union(A, B)
        
    total_operations = 0
    
    # Iterate through all users to find the root of each component.
    for i in range(1, N + 1):
        if dsu.parent[i] == i:
            # i is the representative of a connected component.
            V = dsu.num_nodes[i]
            E = dsu.num_edges[i]
            
            # The number of edges in a complete graph (clique) with V vertices.
            max_possible_edges = V * (V - 1) // 2
            
            # The number of new friendships is the difference.
            operations_for_component = max_possible_edges - E
            
            total_operations += operations_for_component
            
    print(total_operations)

if __name__ == "__main__":
    main()