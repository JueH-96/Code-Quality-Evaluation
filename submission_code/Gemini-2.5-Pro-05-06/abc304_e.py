import sys

class DSU:
    def __init__(self, n):
        # Initialize parent array for 1-based indexing (nodes 1 to N)
        # self.parent[0] and self.set_size[0] will be unused.
        self.parent = list(range(n + 1)) 
        self.set_size = [1] * (n + 1) 

    def find(self, i):
        # Iterative path compression
        path_to_root = []
        curr = i
        while self.parent[curr] != curr: # Traverse up to find the root
            path_to_root.append(curr)
            curr = self.parent[curr]
        
        root = curr # 'curr' is now the root of the set containing i

        # Path compression: make all nodes on the path direct children of the root
        for node_on_path in path_to_root:
            self.parent[node_on_path] = root
        return root

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by size: attach smaller tree under root of larger tree
            if self.set_size[root_i] < self.set_size[root_j]:
                root_i, root_j = root_j, root_i # Ensure root_i's set is larger or equal
            
            self.parent[root_j] = root_i
            self.set_size[root_i] += self.set_size[root_j]
            # Return True indicates a merge happened, False otherwise.
            # This return value is not strictly needed for this problem's logic.
            return True 
        return False

def main():
    N, M = map(int, sys.stdin.readline().split())

    dsu = DSU(N)

    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        dsu.union(u, v)

    K = int(sys.stdin.readline())
    
    forbidden_component_pairs = set()
    for _ in range(K):
        x, y = map(int, sys.stdin.readline().split())
        
        root_x = dsu.find(x)
        root_y = dsu.find(y)
        
        # As per problem statement: "For all i = 1, ..., K, 
        # there is no path connecting vertices x_i and y_i."
        # This implies that in graph G, x_i and y_i are in different components.
        # Thus, root_x will not be equal to root_y.
        
        # Store pairs of component roots in a canonical form, e.g., (min_root, max_root),
        # to ensure that (root_x, root_y) and (root_y, root_x) are treated identically when added to the set.
        if root_x > root_y:
            root_x, root_y = root_y, root_x # Swap to ensure root_x <= root_y
        forbidden_component_pairs.add((root_x, root_y))

    Q = int(sys.stdin.readline())
    # Using a list to store results and printing them all at once is generally faster
    # than printing each result immediately, due to I/O overhead.
    results = [] 
    for _ in range(Q):
        p, q = map(int, sys.stdin.readline().split())
        
        root_p = dsu.find(p)
        root_q = dsu.find(q)
        
        if root_p == root_q:
            # If p and q are already in the same component in G,
            # adding an edge (p,q) does not merge any components.
            # Since G is good, G + (p,q) will also be good.
            results.append("Yes")
        else:
            # If p and q are in different components, C(p) and C(q),
            # adding an edge (p,q) will merge C(p) and C(q).
            # We need to check if this merge makes any forbidden pair (x_j, y_j) connected.
            # This happens if {C(x_j), C(y_j)} is exactly {C(p), C(q)}.
            
            # Convert {root_p, root_q} to canonical form for lookup in the set.
            if root_p > root_q:
                root_p, root_q = root_q, root_p # Swap to ensure root_p <= root_q
            
            if (root_p, root_q) in forbidden_component_pairs:
                # The pair of components being merged {C(p), C(q)} is one of the
                # pairs that are forbidden to be connected. So, G + (p,q) is not good.
                results.append("No")
            else:
                # The merge of C(p) and C(q) does not affect any of the K forbidden pairs.
                # So, G + (p,q) remains good.
                results.append("Yes")
                
    # Print all results, each on a new line.
    sys.stdout.write("
".join(results) + "
")

if __name__ == '__main__':
    main()