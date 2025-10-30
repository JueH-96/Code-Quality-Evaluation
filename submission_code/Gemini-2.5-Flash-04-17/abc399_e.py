import sys

def solve():
    # Read input
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # 1. Consistency check and build target map
    # target map: key = char in S, value = required target char in T
    # This map stores the final character each unique character from the original S must become.
    target = {}
    for i in range(N):
        s_char = S[i]
        t_char = T[i]
        if s_char in target:
            # If character s_char has been seen before, check if the target T[i] is consistent
            # All occurrences of the same character in the original S must map to the same character in T.
            if target[s_char] != t_char:
                # Inconsistent target for the same character from S, impossible transformation
                print(-1)
                return
        else:
            # First time seeing this character from S, record its target
            target[s_char] = t_char

    # 2. Initialize DSU for 'a' through 'z' (0 through 25)
    # We use DSU to track components of characters that must eventually become the same.
    # The structure of the required transformations (c -> target(c)) forms a functional graph.
    # parent[i] is the parent of character with index i (0-25). Initially, each char is its own parent.
    parent = list(range(26))
    # has_cycle[i] is True if the component whose root is index i contains a cycle in the
    # required transformation graph (edges c -> target(c) for c != target(c)).
    has_cycle = [False] * 26

    # Find operation with path compression
    def find(c_ord):
        if parent[c_ord] == c_ord:
            return c_ord
        # Path compression: set parent to the root
        parent[c_ord] = find(parent[c_ord])
        return parent[c_ord]

    # Union operation based on the directed edges c -> target(c)
    def union(u_ord, v_ord):
        root_u = find(u_ord)
        root_v = find(v_ord)

        if root_u != root_v:
            # If u and v are in different sets, merge them.
            # Adding edge u -> v (where target(u) = v) merges the component of v into the component of u.
            # Merge root_v's set into root_u's set (arbitrary choice, can use size/rank for performance).
            parent[root_v] = root_u
            # The new component has a cycle if either of the merged components had a cycle.
            # (This helps maintain cycle status correctly on the root during merges)
            has_cycle[root_u] = has_cycle[root_u] or has_cycle[root_v]
        else:
            # If u and v are already in the same set, adding the edge u -> v creates a cycle
            # within this existing component structure.
            has_cycle[root_u] = True

    # 3. Count required direct transformation edges and perform DSU union
    # The required direct transformation edges are u -> target(u) for all u in S where u != target(u).
    # Each such edge represents a transformation that needs to be performed eventually.
    ops = 0
    # Iterate through all possible lowercase characters 'a' through 'z' (0 through 25)
    for i in range(26):
        char = chr(i + ord('a'))
        
        # Check if this character appeared in S and requires a transformation (c != target(c)).
        # The `target` map only contains keys for characters that were present in S.
        if char in target and target[char] != char:
            # This indicates a required direct transformation edge: char -> target[char]
            # The number of such distinct edges represents the base number of operations.
            ops += 1 
            
            u_ord = ord(char) - ord('a')
            v_ord = ord(target[char]) - ord('a')
            
            # Perform union operation based on this required transformation edge.
            # This groups characters that must eventually become the same.
            # It also detects cycles formed by these required transformations.
            union(u_ord, v_ord)

    # 4. Count components with cycles
    # After processing all relevant edges, count how many distinct components have the has_cycle flag set.
    # Each component with a cycle requires one extra operation (e.g., using a temporary character)
    # beyond the simple merges represented by the edges within that component.
    cycle_components = 0
    # Iterate through all characters 'a' through 'z' (0 through 25)
    for i in range(26):
        # If this character index 'i' is the root of its component (parent[i] == i)
        # and that component has been marked as having a cycle (has_cycle[i] is True for the root)
        if parent[i] == i and has_cycle[i]:
             cycle_components += 1

    # The minimum number of operations is the sum of:
    # - The number of required direct transformation edges (ops).
    # - The number of components that contain a cycle (cycle_components).
    # Each edge corresponds to one merge operation. A cycle in a component requires an extra operation
    # to break the cycle dependency (e.g., using a temporary character), applied once per cycle component.
    print(ops + cycle_components)

solve()