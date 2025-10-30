import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # mapping[c_code] stores the character that c_code MUST map to finally.
    # Initialize with -1 to indicate 'no mapping yet'.
    mapping = [-1] * 26

    # Build the necessary mapping and check for inconsistencies
    for i in range(N):
        s_c_code = ord(S[i]) - ord('a')
        t_c_code = ord(T[i]) - ord('a')

        if mapping[s_c_code] == -1:
            # First time seeing this source character, assign its target.
            mapping[s_c_code] = t_c_code
        elif mapping[s_c_code] != t_c_code:
            # This source character was already assigned a different target. Impossible.
            print(-1)
            return

    # Create the final char_map for graph traversal.
    # If a character had no explicit mapping from S to T, it implicitly maps to itself.
    char_map = [0] * 26
    for i in range(26):
        if mapping[i] == -1:
            char_map[i] = i # Character maps to itself
        else:
            char_map[i] = mapping[i]

    ops = 0
    visited = [False] * 26 # Global visited array to track processed nodes
    on_recursion_stack = [False] * 26 # To detect cycles within the current DFS path

    # Iterate through all characters 'a' through 'z'
    for i in range(26):
        # If the character is already visited or it maps to itself (no change needed)
        if visited[i] or char_map[i] == i:
            continue

        # Start DFS traversal from `i`
        curr = i
        path_nodes = [] # Stores nodes in the current DFS path for cycle detection

        while char_map[curr] != curr and not visited[curr]:
            visited[curr] = True
            on_recursion_stack[curr] = True
            path_nodes.append(curr)
            curr = char_map[curr]
        
        # After the loop, `curr` is either a self-looping node,
        # or a node that was already visited (either in a previous component or in the current path).

        # Each node in `path_nodes` (that isn't a self-loop) needs an operation.
        # This is `len(path_nodes)` operations.
        ops += len(path_nodes)
        
        # Check for a cycle: if `curr` is currently in the recursion stack, it's a cycle.
        if on_recursion_stack[curr]:
            ops += 1 # Add 1 for breaking the cycle

        # Clear on_recursion_stack flags for nodes processed in this DFS traversal.
        for node in path_nodes:
            on_recursion_stack[node] = False

    print(ops)

solve()