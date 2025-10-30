# YOUR CODE HERE
import sys

# Increase recursion depth limit for safety, although the main logic is iterative.
# This might be relevant for deep function call chains if Python interpreter internally uses recursion for some operations,
# or if the functional graph has very long paths.
try:
    # Set a large enough recursion depth considering N can be up to 2*10^5
    # Adding a small buffer just in case.
    sys.setrecursionlimit(200010 + 10) 
except Exception as e: 
    # In some environments (like restricted online judges), setting recursion depth might fail.
    # Silently ignore the exception and proceed with the default limit.
    pass 

def solve():
    # Read N (length of sequences) and K (number of operations)
    N, K = map(int, sys.stdin.readline().split())
    
    # Read sequence X (1-based indices)
    X = list(map(int, sys.stdin.readline().split()))
    # Convert X to 0-based indices for use with Python lists/arrays
    X_adj = [x - 1 for x in X] 

    # Read initial sequence A (values)
    A = list(map(int, sys.stdin.readline().split()))

    # Base case: If K=0, no operations are performed. Print the initial sequence A and exit.
    if K == 0:
        print(*(A))
        return

    # Define the transformation function f: index j -> X_adj[j]
    # The operation replaces A with B such that B_i = A_{X_i}. With 0-based indexing, B_i = A_{X_adj[i]}.
    # Let f(i) = X_adj[i]. After 1 step, the value at index `i` comes from index `f(i)` of the previous sequence.
    # After k steps, the value at index `i` comes from index `f^k(i)` of the initial sequence A^{(0)}.
    # So, A^{(k)}_i = A^{(0)}_{f^k(i)}.
    # Our goal is to compute f^K(i) for all i = 0, ..., N-1.
    def f(j):
        # Given an index j, return the index from which the value at j will be sourced in the next step.
        return X_adj[j]

    # final_pos[i] will store the index p such that the final value A'_i = A^{(0)}_p.
    # This index p is precisely f^K(i). Initialize with -1 indicating 'not computed'.
    final_pos = [-1] * N
    
    # visited_step[i] serves two purposes:
    # 1. Marks if a node `i` has been visited during the *current* path trace starting from some root node.
    # 2. Stores the step number (0-indexed) when node `i` was first visited in this trace.
    # This is used to detect cycles and determine the cycle start point and length.
    # It is reset to -1 for the relevant nodes after each component trace is fully processed.
    visited_step = [-1] * N 

    # Iterate through all possible starting nodes (indices 0 to N-1).
    # Each iteration processes one connected component of the functional graph if not already processed.
    for i in range(N):
        # If the final position for node `i` has already been computed (i.e., it was part of a previously processed component), skip it.
        if final_pos[i] != -1:
            continue

        # Start a new path trace from node `i`.
        path = [] # Stores the sequence of nodes visited in the current path trace.
        curr = i  # `curr` is the current node being visited in the trace.
        step = 0  # `step` counts the number of steps taken in the current trace (length of path so far - 1).
        
        # Trace the path: curr -> f(curr) -> f(f(curr)) -> ...
        # The loop continues as long as we visit nodes not previously seen *in this specific trace*.
        while visited_step[curr] == -1:
            # Mark node `curr` as visited at step `step` in the current trace.
            visited_step[curr] = step
            # Add `curr` to the path history.
            path.append(curr)
            # Move to the next node using the function f.
            curr = f(curr)
            # Increment step counter.
            step += 1
            
            # Check if the new `curr` node has already been computed (`final_pos != -1`).
            # If so, the current path merges into an already fully processed component.
            # We can stop the trace here and compute the final positions for the nodes in `path`
            # by determining how many steps they take to reach `curr` and then how many steps `curr` takes
            # to reach its final destination f^K(curr). This is complex.
            # A simpler approach is to just continue the trace. Eventually, it must enter a cycle.
            # The cycle might be fully contained within the already processed component or involve nodes from the current path.
            # The cycle detection logic based on `visited_step` will handle this correctly.
            # Let's stick to detecting cycles purely based on `visited_step`.

            # Check if the new `curr` node has been visited before *in this current trace*.
            if visited_step[curr] != -1:
                # Cycle detected! `curr` was previously visited in this trace.
                # `p` is the step number when `curr` was first visited (start of the cycle in the path).
                p = visited_step[curr] 
                # `k` is the current step number. The path has length `k` (indices 0 to k-1).
                k = step 
                # `c` is the length of the cycle.
                c = k - p 

                # Compute final positions `f^K(j)` for all nodes `j` in the discovered path `path`.
                for m in range(k):
                    j = path[m] # The node visited at step `m`.
                    
                    # If final_pos[j] was already computed (e.g., from trace of another component merging here), skip re-computation.
                    if final_pos[j] != -1: 
                        continue

                    # Case 1: Node `j` is in the preperiod path (part of the path before the cycle). `m < p`.
                    if m < p: 
                        # If K steps are fewer than the steps needed for `j` to reach the cycle start `path[p]`.
                        if K < p - m:
                             # The final position after K steps is simply the node at index `m+K` in the path.
                            final_pos[j] = path[m + K]
                        else: 
                            # Node `j` reaches the cycle start `path[p]` in `p-m` steps.
                            # It needs to take `K - (p-m)` more steps within the cycle.
                            # The position within the cycle (0 to c-1) is determined by these remaining steps modulo `c`.
                            idx_in_cycle = (K - (p - m)) % c
                            # The final node reached is at index `p + idx_in_cycle` in the `path` list.
                            final_pos[j] = path[p + idx_in_cycle]
                    
                    # Case 2: Node `j` is on the cycle. `m >= p`.
                    else: 
                        # Node `j` is `path[m]`. Its position relative to the cycle start `path[p]` is `m-p` (0-indexed within cycle nodes).
                        # After K steps starting from this position within the cycle, its relative position will be `(m - p + K) % c`.
                        idx_in_cycle = (m - p + K) % c
                        # The final node reached is at index `p + idx_in_cycle` in the `path` list.
                        final_pos[j] = path[p + idx_in_cycle]

                # Break the while loop since cycle detected and path processed.
                break
        
        # After processing the path (either cycle detected or merged), reset `visited_step` for nodes involved in this trace.
        # This ensures they are correctly marked as unvisited for subsequent traces starting from different root nodes.
        # Resetting is needed because `visited_step` tracks visits *within a single trace*.
        current_path_len = len(path) # Get the actual length of path traced in this component run.
        for node_idx in range(current_path_len): 
            node_in_path = path[node_idx]
            visited_step[node_in_path] = -1
        # Need to also potentially reset `curr` if the loop terminated because `visited_step[curr]` was non -1.
        # This `curr` node is `path[p]`, which is included in the loop above if path is non-empty.
        # If the path is empty (i=0, final_pos[0] != -1), loop doesn't run. Resetting is not needed.
        # If path has length > 0, the loop handles resetting nodes in path.


    # After iterating through all i and computing final_pos[i] = f^K(i) for all i,
    # construct the final sequence A'.
    final_A = [0] * N
    for i in range(N):
        # The value at index `i` in the final sequence A' comes from the initial sequence A
        # at the index `final_pos[i]`.
        final_A[i] = A[final_pos[i]]

    # Print the final sequence elements separated by spaces.
    print(*(final_A))

# Execute the solver function defined above.
solve()