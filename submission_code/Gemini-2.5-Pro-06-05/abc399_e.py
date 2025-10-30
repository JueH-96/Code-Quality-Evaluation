import sys
from collections import defaultdict

# It is good practice to increase the recursion limit for deep recursion,
# though for a DSU on 26 elements with path compression, it's not strictly necessary.
sys.setrecursionlimit(2 * 10**5 + 5)

class DSU:
    """Disjoint Set Union (Union-Find) data structure."""
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        """Find the representative of the set containing element i."""
        if self.parent[i] == i:
            return i
        # Path compression for optimization
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Merge the sets containing elements i and j."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            return True
        return False

def solve():
    """
    Main function to solve the problem.
    """
    # 1. Read input and build the character mapping function
    try:
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
        T = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handle cases with no input
        return

    # m_map[i] stores the character that character 'a'+i must become.
    # Initialize with -1 to indicate not yet mapped.
    m_map = [-1] * 26
    
    for i in range(N):
        s_char_code = ord(S[i]) - ord('a')
        t_char_code = ord(T[i]) - ord('a')
        
        if m_map[s_char_code] == -1:
            m_map[s_char_code] = t_char_code
        elif m_map[s_char_code] != t_char_code:
            # Inconsistent mapping required for a character. Impossible.
            # e.g., 'a' needs to become both 'b' and 'c'.
            print(-1)
            return

    # For characters not present in S, they map to themselves.
    for i in range(26):
        if m_map[i] == -1:
            m_map[i] = i

    # The mapping defines a functional graph on the 26 characters.
    # Nodes are 0-25 ('a'-'z'), edges are i -> m_map[i].
    
    # 2. Use DSU to find the connected components of the graph (viewed as undirected).
    dsu = DSU(26)
    for i in range(26):
        dsu.union(i, m_map[i])

    # 3. For each component, count its number of nodes and edges.
    nodes_in_comp = defaultdict(int)
    edges_in_comp = defaultdict(int)
    
    for i in range(26):
        root = dsu.find(i)
        nodes_in_comp[root] += 1
        if m_map[i] != i:
            edges_in_comp[root] += 1
            
    # 4. Calculate the minimum number of operations based on component structure.
    total_ops = 0
    has_cycle_component = False
    
    unique_roots = {dsu.find(i) for i in range(26)}
    
    for root in unique_roots:
        nodes = nodes_in_comp[root]
        edges = edges_in_comp[root]
        
        # A component is tree-like if #edges = #nodes - 1.
        # It contains a cycle if #edges = #nodes.
        if edges > 0:  # Only consider components with actual transformations
            if edges == nodes:
                has_cycle_component = True
                total_ops += edges + 1
            elif edges == nodes - 1:
                total_ops += edges

    # 5. Handle the special case where a cycle exists but no temporary
    #    character is available.
    num_nontrivial_maps = sum(1 for i in range(26) if m_map[i] != i)
    
    if has_cycle_component and num_nontrivial_maps == 26:
        # All 26 characters are part of non-trivial transformations, forming cycles.
        # There's no "free" character to use as a temporary placeholder.
        print(-1)
    else:
        print(total_ops)

solve()