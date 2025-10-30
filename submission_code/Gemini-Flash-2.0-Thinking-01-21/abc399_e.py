import sys

def solve():
    # Increase recursion depth if needed, although the iterative DFS avoids deep stacks
    # sys.setrecursionlimit(2000) # Not strictly necessary for this iterative approach

    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Step 1: Build the required mapping and check consistency
    # target[i] stores the character index 'a'+i must map to
    # Use -1 initially to indicate no mapping requirement seen yet for this character from S/T comparison
    target = [-1] * 26

    for i in range(N):
        u_idx = ord(S[i]) - ord('a')
        v_idx = ord(T[i]) - ord('a')

        if target[u_idx] == -1:
            # First time setting a mapping requirement for u_idx based on S[i]/T[i]
            target[u_idx] = v_idx
        elif target[u_idx] != v_idx:
            # Inconsistent mapping required for u_idx
            print(-1)
            return

    # Fill in self-mappings for characters that appeared in S but didn't get a non-self mapping
    # and for characters that did not appear in S at all.
    # Any character i for which target[i] is still -1 means no S[j] == chr(ord('a')+i) existed.
    # These characters don't have an explicit target requirement from S/T, they can effectively map to themselves.
    for i in range(26):
        if target[i] == -1:
            target[i] = i # Character maps to itself if no requirement forced it otherwise

    # Step 2: Count characters that must change their value
    # These are the characters 'u' for which target[u] != u.
    n_changed_chars = 0
    for i in range(26):
        if target[i] != i:
            n_changed_chars += 1

    # Step 3: Detect cycles of length > 1 in the functional graph i -> target[i]
    # Use states for iterative DFS: 0: unvisited, 1: visiting (in current DFS path), 2: visited (component processed)
    n_cycles = 0
    state = [0] * 26

    for i in range(26):
        if state[i] == 0:
            curr = i
            path_nodes = [] # Indices in the current DFS path

            # Traverse the component starting from i
            while state[curr] == 0:
                state[curr] = 1 # Mark as visiting
                path_nodes.append(curr)
                curr = target[curr] # Move to the next character in the required chain

            # After the loop, curr is either 'visiting' (state 1, cycle detected within current path)
            # or 'visited' (state 2, path merges into a previously processed component)
            if state[curr] == 1: # curr is in the current path -> Cycle detected
                # Find where curr first appeared in path_nodes to determine the cycle length
                try:
                    cycle_start_index = path_nodes.index(curr)
                    cycle_length = len(path_nodes) - cycle_start_index
                    
                    # A cycle of length > 1 needs an extra operation.
                    # A cycle of length 1 (self-loop i -> i) does not contribute to n_cycles.
                    if cycle_length > 1:
                        n_cycles += 1
                except ValueError:
                     # This case should theoretically not be reached in a functional graph traversal
                     # from an unvisited node if state[curr] == 1 implies curr is in path_nodes.
                     # If state[curr] is 1, it means 'curr' is in the current recursion stack/path.
                     pass # Defensive programming, unlikely needed

            # Mark all nodes in the current path as fully visited (state 2)
            # Even if a cycle was found, or the path merged, all nodes encountered are processed from this starting point i.
            for node_idx in path_nodes:
                 state[node_idx] = 2

    # The minimum number of operations is the sum of operations for each component.
    # A component is either a set of paths leading to a self-loop, or a set of paths leading to a cycle.
    # For a path component (ending in a self-loop): the number of ops equals the number of nodes 'u' in the component where target[u] != u.
    # For a cycle component: the number of ops equals the number of nodes 'u' in the cycle where target[u] != u (which is the cycle length if > 1) PLUS one extra operation.
    # Paths leading into a cycle: The nodes in the path also need changing.
    # The total number of operations sums up the 'cost' for each character needing change (`n_changed_chars`)
    # and adds an additional operation for each non-trivial cycle (`n_cycles`).
    print(n_changed_chars + n_cycles)

solve()