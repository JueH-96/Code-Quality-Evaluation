import sys

def solve():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # 1. Consistency check and build required mapping
    # target_map maps a character c in S to the character it must become in T.
    # For consistency, if c appears multiple times in S, the corresponding characters in T must be the same.
    target_map = {}
    for i in range(N):
        s_char = S[i]
        t_char = T[i]
        if s_char in target_map:
            if target_map[s_char] != t_char:
                # Inconsistent requirement: same character in S must map to different characters in T. Impossible.
                print(-1)
                return
        else:
            target_map[s_char] = t_char

    # Build the functional graph where an edge u -> v exists if character u (present in S)
    # must become character v (in T) and u != v.
    # Use index 0-25 for 'a'-'z'.
    # graph[i] = j means character 'a'+i must become character 'a'+j.
    # Initialize with -1 indicating no required transformation edge from this character.
    graph = [-1] * 26
    initial_ops = 0 # Count of required transformations u -> v where u != v

    # We only build edges for characters present in S that need to change (u != target_map[u]).
    for char_s, char_t in target_map.items():
        if char_s != char_t:
            u_idx = ord(char_s) - ord('a')
            v_idx = ord(char_t) - ord('a')
            graph[u_idx] = v_idx
            initial_ops += 1 # Each such required transformation corresponds to at least one conceptual step

    # 2. Count cycles > 1 in the functional graph using DFS.
    # Cycles of length > 1 indicate a circular dependency (e.g., a->b, b->c, c->a).
    # Breaking such a cycle requires an extra operation.
    visited = [0] * 26 # 0: unvisited, 1: visiting (on current stack), 2: visited
    on_stack = [False] * 26 # True if node is currently on the recursion stack
    cycle_count = 0

    # DFS function to traverse the functional graph and detect cycles.
    def dfs(u_idx):
        nonlocal cycle_count
        visited[u_idx] = 1 # Mark node as visiting
        on_stack[u_idx] = True # Add node to recursion stack

        v_idx = graph[u_idx] # Get the character u_idx must become

        if v_idx != -1: # Check if there is a required transformation edge from u_idx
            if visited[v_idx] == 0: # Successor not visited
                dfs(v_idx)
            elif visited[v_idx] == 1: # Successor is currently visiting (on recursion stack) -> Cycle detected
                 # The edge u_idx -> v_idx closes a cycle where v_idx is an ancestor of u_idx.
                 # If u_idx == v_idx, it's a self-loop (length 1), which doesn't require an extra operation.
                 # If u_idx != v_idx, it's a cycle of length > 1, requiring one extra operation to resolve the circular dependency.
                 # Each cycle of length > 1 will have exactly one back-edge discovered during the DFS traversal of the functional graph forest,
                 # ensuring each such cycle is counted exactly once.
                 if u_idx != v_idx:
                     cycle_count += 1

        on_stack[u_idx] = False # Remove node from recursion stack
        visited[u_idx] = 2 # Mark node as fully visited

    # Iterate through all possible starting nodes for DFS ('a' through 'z')
    # We only need to start DFS from nodes that are sources of required transformations
    # (i.e., characters u in S such that u != target_map[u]), as other nodes
    # won't be part of a cycle involving these transformations, unless reached from such a source.
    # However, iterating through all 26 nodes and checking visited state ensures all components
    # of the functional graph derived from target_map are visited.
    for i in range(26):
         # Start DFS if the node has an outgoing edge in the functional graph (is a source u with u != target[u])
         # and hasn't been fully visited yet. Nodes with graph[i] == -1 (no required transformation edge)
         # or nodes where the requirement was u -> u (filtered when building graph) cannot initiate
         # a cycle > 1 in this specific transformation graph structure.
         # Iterating 0-25 and checking visited handles all reachable nodes correctly.
         if visited[i] == 0:
             # Only components containing nodes with graph[i] != -1 matter for cycle detection
             # related to required transformations. But starting DFS from any unvisited node is standard.
             # The cycle count logic correctly filters for cycles formed by graph edges.
             dfs(i)

    # The minimum number of operations is the sum of:
    # - The number of distinct required transformations u -> v where u != v (initial_ops).
    #   Each such transformation effectively costs one operation unless it's part of a chain/merge.
    # - One extra operation for each cycle of length > 1 in the functional graph formed by these transformations.
    #   Cycles require additional steps to resolve the circular dependency.
    print(initial_ops + cycle_count)

solve()