# YOUR CODE HERE
import sys

# It is good practice to set a higher recursion limit for problems that might
# involve deep recursion, although for a DSU with path compression, the actual
# recursion depth will be very small.
sys.setrecursionlimit(2 * 10**5 + 5)

# For performance, especially with large inputs, it's better to use
# sys.stdin.readline instead of the default input().
input = sys.stdin.readline

class DSU:
    """
    An implementation of the Disjoint Set Union (DSU) data structure, also known
    as Union-Find. It is used to keep track of a partition of a set of elements
    into a number of disjoint, non-overlapping subsets.
    
    This implementation uses two key optimizations:
    1. Path Compression: Flattens the structure of the tree whenever `find` is
       called, making future operations faster.
    2. Union by Size: When merging two sets, the smaller tree is attached to the
       root of the larger tree, keeping the trees balanced and shallow.
    
    The vertices are assumed to be 1-indexed from 1 to n.
    """
    def __init__(self, n):
        # parent[i] stores the parent of element i.
        # Initially, each element is in its own set, so it's its own parent.
        self.parent = list(range(n + 1))
        # size[i] stores the size of the set rooted at i.
        # Initially, each set has a size of 1.
        self.size = [1] * (n + 1)

    def find(self, i):
        """
        Finds the representative (or root) of the set containing element i.
        Applies path compression during the find operation.
        """
        if self.parent[i] == i:
            return i
        # Recursively find the root and set the parent of i directly to it.
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """
        Merges the two sets that elements i and j belong to.
        Applies the union-by-size optimization.
        """
        root_i = self.find(i)
        root_j = self.find(j)
        
        if root_i != root_j:
            # Merge the smaller set into the larger one.
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]

def solve():
    """
    This function contains the main logic for solving the problem.
    """
    # Reading the first line can result in an empty string if the input is empty.
    line = input()
    if not line:
        return
    N, M = map(int, line.split())

    # --- Step 1: Determine the connected components of the initial graph G ---
    # We use a DSU data structure to efficiently find the connected components.
    # After processing all M edges, dsu.find(v) will return a unique
    # representative for the component containing vertex v.
    dsu = DSU(N)
    for _ in range(M):
        u, v = map(int, input().split())
        dsu.union(u, v)

    # --- Step 2: Identify "forbidden" pairs of components ---
    # A graph is "good" if for all given pairs (x_i, y_i), there is no path.
    # This implies x_i and y_i are in different connected components.
    # We pre-calculate the pairs of components that must remain separate.
    K = int(input())
    forbidden_pairs = set()
    for _ in range(K):
        x, y = map(int, input().split())
        root_x = dsu.find(x)
        root_y = dsu.find(y)
        
        # To treat the pair (a, b) the same as (b, a), we store them in a
        # canonical order, for example, with the smaller root first.
        # This allows for consistent lookups in our `forbidden_pairs` set.
        if root_x > root_y:
            root_x, root_y = root_y, root_x
        forbidden_pairs.add((root_x, root_y))

    # --- Step 3: Process the Q independent queries ---
    # For each query, we consider adding an edge (p, q). This action would
    # merge the component of p with the component of q.
    # The graph becomes "not good" if this merge operation connects a pair of
    # vertices (x_i, y_i) that should remain disconnected. This happens if the
    # pair of components {component_of_p, component_of_q} is one of the
    # pairs we identified as forbidden.
    Q = int(input())
    results = []
    for _ in range(Q):
        p, q = map(int, input().split())
        
        root_p = dsu.find(p)
        root_q = dsu.find(q)
        
        # Again, use the canonical representation for the lookup.
        if root_p > root_q:
            root_p, root_q = root_q, root_p
            
        # Check if merging the components of p and q is a forbidden action.
        # If the pair of roots is in our set, the graph becomes "not good".
        # Note: If p and q are already in the same component, root_p == root_q.
        # The pair (root_p, root_p) will not be in `forbidden_pairs` because
        # the problem guarantees x_i != y_i and they are initially disconnected,
        # so their roots are different. Thus, this case correctly yields "Yes".
        if (root_p, root_q) in forbidden_pairs:
            results.append("No")
        else:
            results.append("Yes")
            
    # --- Step 4: Print all the collected results ---
    print("
".join(results))

# Execute the main solver function.
solve()