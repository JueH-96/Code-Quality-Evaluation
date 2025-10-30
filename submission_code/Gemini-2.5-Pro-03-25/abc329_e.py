# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep DFS
# The number of nodes in the graph is N-M+1, which can be up to 200000.
# A deep path could exceed default limits.
try:
    # Set a large recursion depth limit, but respect potential system limits
    # Usually 2*10^5 is sufficient for N=2*10^5
    # Using a slightly larger value for safety margin.
    sys.setrecursionlimit(2 * 10**5 + 100) 
except Exception: # Catch broadly, specific exceptions might differ across systems
    # If setting limit fails, code might fail on deep recursion cases.
    pass


def solve():
    # Read input values N, M
    N, M = map(int, sys.stdin.readline().split())
    # Read strings S and T
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # E[k] stores a list of operation start indices i such that
    # applying the operation starting at i covers position k (i.e., i <= k < i+M)
    # AND the character placed at position k matches S[k] (i.e., T[k-i] == S[k]).
    # This list represents all operations that could potentially be the *last* operation
    # responsible for setting the character S[k].
    E = [[] for _ in range(N)]
    
    # Calculate the number of possible starting indices for the operation.
    # Operations can start from index 0 up to N-M.
    possible_op_indices = N - M + 1

    # Precompute T characters into a list for potentially faster access,
    # although string indexing in Python is typically efficient.
    T_chars = list(T)

    # Populate the E lists for each position k.
    # Iterate through all possible operation start indices i.
    # Time complexity: O((N-M+1) * M) which simplifies to O(NM)
    # Since M is small (<=5), this is efficient enough (approx O(N)).
    for i in range(possible_op_indices):
        # The operation starting at i covers indices k from i to i+M-1.
        for k_offset in range(M):
            k = i + k_offset
            # Check boundary: k must be a valid index within string S [0, N-1].
            # This condition is implicitly true since i <= N-M and k_offset < M implies i+k_offset <= N-M+M-1 = N-1.
            # So k < N is always true if loop bounds are correct. No explicit check needed unless N=0 edge case? N>=1 given.
            
            # Check if the character T provides at this relative offset matches the character in S at absolute position k.
            if T_chars[k_offset] == S[k]:
                 # If they match, operation i is a valid candidate for setting S[k].
                 E[k].append(i) 

    # Initial check: Is it possible to form S at all?
    # If for any position k, E[k] is empty, it means no operation can produce the required character S[k].
    # In this case, it's impossible to form S.
    # Time complexity: O(N)
    for k in range(N):
        if not E[k]:
            print("No")
            return

    # Build the dependency graph based on "forced" positions.
    # A position k is "forced" if only one operation i (i.e., len(E[k]) == 1) can produce S[k].
    # If k is forced by i, then operation i *must* be the last operation covering k.
    # This imposes constraints on the relative order of operations.
    # adj[i] stores a list of operations that must happen AFTER operation i.
    # Equivalently, we model edges i_prime -> i meaning i_prime must happen BEFORE i.
    adj = [[] for _ in range(possible_op_indices)]
    
    # Use a set to keep track of edges already added to avoid duplicates.
    # This can slightly improve performance for DFS on dense graphs, though not strictly necessary for correctness.
    added_edges = set()

    # Iterate through all positions k to identify forced constraints and build the graph.
    # Time complexity: O(N * M) because the inner loop iterates at most M times.
    # Since M <= 5, this is effectively O(N).
    for k in range(N):
        # Check if position k is forced (only one possible operation i).
        if len(E[k]) == 1:
            i = E[k][0] # The unique operation index that *must* be the last one setting S[k].

            # Consider any other operation i_prime that also covers position k.
            # The range of such i_prime is from max(0, k-M+1) up to k.
            # Iterate through potential conflicting operations i_prime.
            # The upper bound is min(possible_op_indices, k + 1) to ensure i_prime is a valid op index
            # and i_prime <= k.
            for i_prime in range(max(0, k - M + 1), min(possible_op_indices, k + 1)):
                 # Skip comparing the operation with itself.
                 if i_prime == i:
                     continue

                 # Check if operation i_prime actually covers k.
                 # This check is implicitly handled by the loop range `max(0, k-M+1) <= i_prime <= k`.
                 # Combined with `i_prime < i_prime + M`, this implies `k < i_prime + M`.
                 
                 # Check if operation i_prime conflicts with S[k], meaning it would set the wrong character.
                 # Calculate the index within T that operation i_prime places at k.
                 k_offset_prime = k - i_prime 
                 
                 # Boundary check for k_offset_prime (0 <= k_offset_prime < M) is guaranteed because:
                 # i_prime <= k -> k - i_prime >= 0
                 # k < i_prime + M -> k - i_prime < M
                 
                 if T_chars[k_offset_prime] != S[k]:
                     # If i_prime covers k BUT sets the WRONG character (T[k-i'] != S[k]),
                     # then i_prime must happen strictly BEFORE operation i to allow i to be the last one fixing S[k].
                     # This dependency translates to a directed edge: i_prime -> i.
                     edge = (i_prime, i)
                     # Add the edge only if it hasn't been added before.
                     if edge not in added_edges:
                         adj[i_prime].append(i)
                         added_edges.add(edge)


    # Detect cycles in the constructed dependency graph using Depth First Search (DFS).
    # A cycle implies contradictory ordering constraints, making it impossible to satisfy all forced conditions.
    # State representation for DFS nodes:
    # 0: unvisited
    # 1: visiting (currently in the recursion stack)
    # 2: visited (finished exploring all paths from this node) - using this state can optimize repeated visits.
    visited = [0] * possible_op_indices 
    recursion_stack = [0] * possible_op_indices
    
    has_cycle = False

    # Helper function for DFS-based cycle detection.
    # Returns True if a cycle is detected starting from 'node', False otherwise.
    def detect_cycle_util(node):
        visited[node] = 1 # Mark as visiting
        recursion_stack[node] = 1 # Add to current recursion path

        # Explore neighbors
        for neighbor in adj[node]:
            if visited[neighbor] == 0: # If neighbor is unvisited
                 if detect_cycle_util(neighbor): # Recurse
                     return True # Cycle found downstream
            elif recursion_stack[neighbor] == 1: # If neighbor is already in the recursion stack
                 # This indicates a back edge to an ancestor, hence a cycle.
                 return True 

        # Backtrack: remove node from recursion stack after exploring all its paths.
        recursion_stack[node] = 0 
        visited[node] = 2 # Mark node as fully visited.
        return False # No cycle found from this node.

    # Iterate through all nodes (possible operation indices) to check for cycles.
    # This ensures all connected components of the graph are checked.
    # Total time complexity for cycle detection: O(V+E), where V = N-M+1 (nodes), E = number of edges.
    # Since E can be up to O(NM), total time is O(N + NM), which simplifies to O(NM).
    for node in range(possible_op_indices):
        if visited[node] == 0: # Start DFS only if node hasn't been visited yet.
            if detect_cycle_util(node):
                has_cycle = True
                break # Cycle found, no need to check further nodes.

    # Final decision based on cycle detection.
    if has_cycle:
        # A cycle means impossible ordering, so print "No".
        print("No")
    else:
        # If the graph is a Directed Acyclic Graph (DAG), it implies there's no contradiction
        # in the required operation orderings. It has been shown (or assumed based on problem structure)
        # that if the constraints from forced positions are consistent (DAG), then a valid sequence exists.
        print("Yes")

# Execute the solve function
solve()