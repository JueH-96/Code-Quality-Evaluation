import sys

# DSU (Disjoint Set Union) class
class DSU:
    def __init__(self, n):
        # parent[i] stores the parent of element i. If parent[i] == i, then i is the root of its set.
        # 1-indexed, so size n+1
        self.parent = list(range(n + 1))
        # size[i] stores the size of the set if i is the root. Used for union by size heuristic.
        self.size = [1] * (n + 1) 

    # Find operation with path compression.
    def find(self, i):
        if self.parent[i] == i:
            return i
        # Path compression: make current node directly point to the root
        self.parent[i] = self.find(self.parent[i]) 
        return self.parent[i]

    # Union operation with union by size heuristic.
    # Returns True if a merge happened, False otherwise (i.e., i and j were already in the same set).
    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)

        if root_i != root_j:
            # Union by size: attach the smaller tree under the root of the larger tree
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i # Swap roots to ensure root_i is the larger tree root
            
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            return True
        return False

def solve():
    # Use sys.stdin.readline for faster input operations than input()
    input = sys.stdin.readline
    N, M = map(int, input().split())

    dsu = DSU(N)

    # Process M initial edges to build the connected components of graph G
    for _ in range(M):
        u, v = map(int, input().split())
        dsu.union(u, v)

    K = int(input())
    # This set will store canonical pairs of component roots that are forbidden to be connected.
    # A canonical pair is (min(root_A, root_B), max(root_A, root_B)) to handle (A,B) and (B,A) equivalently.
    forbidden_component_pairs = set()

    # Precompute forbidden component pairs
    for _ in range(K):
        x, y = map(int, input().split())
        root_x = dsu.find(x)
        root_y = dsu.find(y)
        
        # According to the problem statement, for all (x_i, y_i), there is no path connecting them in G.
        # This implies root_x must be different from root_y in the initial graph G.
        # We add the canonical pair of their component roots to the forbidden set.
        forbidden_component_pairs.add(tuple(sorted((root_x, root_y))))

    Q = int(input())
    results = [] # To store "Yes" or "No" for each query

    # Process Q queries
    for _ in range(Q):
        p, q = map(int, input().split())
        root_p = dsu.find(p)
        root_q = dsu.find(q)

        if root_p == root_q:
            # If p and q are already in the same connected component in G,
            # adding edge (p,q) does not change existing connectivity (they were already connected).
            # Therefore, G' is still good.
            results.append("Yes")
        else:
            # If p and q are in different components, adding edge (p,q) merges their components.
            # Check if this merge would connect any forbidden (x_i, y_i) pair.
            # This happens if (root_p, root_q) is one of the forbidden_component_pairs.
            
            # Create the canonical pair for (root_p, root_q) for set lookup
            canonical_pair = tuple(sorted((root_p, root_q)))
            
            if canonical_pair in forbidden_component_pairs:
                # If the merged components correspond to a forbidden component pair,
                # then a forbidden (x_i, y_i) pair now has a path. G' is not good.
                results.append("No")
            else:
                # Otherwise, no forbidden pair became connected by this merge. G' is still good.
                results.append("Yes")
    
    # Print all results separated by newlines using sys.stdout.write for faster output
    sys.stdout.write("
".join(results) + "
")

# This is the main entry point for the script
if __name__ == '__main__':
    solve()