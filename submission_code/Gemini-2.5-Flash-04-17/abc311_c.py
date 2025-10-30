import sys

# Read N
N = int(sys.stdin.readline())

# Read A_1, A_2, ..., A_N
# Use 1-based indexing for A, so A[i] is the destination from vertex i
A = [0] * (N + 1)
a_values = list(map(int, sys.stdin.readline().split()))
for i in range(N):
    A[i + 1] = a_values[i]

# state: 0 - unvisited, 1 - visiting (in current path trace), 2 - visited (fully processed component)
# time: stores the step number (1-indexed) when a node was first visited in the current trace (state 1).
# time[u] is only valid when state[u] == 1 for the current trace.
state = [0] * (N + 1)
time = [0] * (N + 1) # Stores step number for state=1 nodes

# Iterate through each vertex as a potential starting point for trace
# We are guaranteed to find a cycle, so we only need to find one.
# Starting a trace from any unvisited node is guaranteed to eventually hit a cycle
# (either within its component's tree leading to its cycle, or being part of the cycle itself).
# The loop over start_node ensures we explore different components if necessary until a cycle is hit.
for start_node in range(1, N + 1):
    # Only start a new trace if this node hasn't been processed fully (state 0)
    if state[start_node] == 0:
        current = start_node
        path = [] # Stores the nodes visited in the current trace path (nodes marked state 1)
        step = 0 # Step counter for the current trace, starts before the first node is processed

        # Trace while the current node is unvisited (state 0)
        # Nodes encountered will be added to 'path', marked state 1, and their visit time recorded
        while state[current] == 0:
            step += 1
            state[current] = 1  # Mark as visiting (part of the current trace)
            time[current] = step # Record visit step number (1-indexed)
            path.append(current)
            current = A[current] # Follow the edge

        # After the loop, state[current] is either 1 (visiting - cycle found) or 2 (visited - hit explored component)
        
        if state[current] == 1:
            # Found a cycle. 'current' is the node where the cycle closes.
            # It must be one of the nodes currently marked state 1 in the 'path'.
            # The step it was first visited in this trace is time[current].
            cycle_start_step = time[current]
            
            # The nodes visited from step cycle_start_step onwards form the cycle.
            # 'path' was built sequentially: path[0] is step 1, path[1] is step 2, ..., path[s-1] is step s.
            # So, the node visited at step cycle_start_step is at index cycle_start_step - 1 in 'path'.
            cycle_start_index_in_path = cycle_start_step - 1
            cycle = path[cycle_start_index_in_path:]
            
            # Output the cycle
            print(len(cycle))
            print(*cycle)
            
            # Found a cycle, terminate the program
            sys.exit()

        # If state[current] == 2:
        # The trace path 'path' led to an already visited component 'current' (marked state 2 in a previous trace).
        # The nodes in 'path' were marked state 1 during this trace.
        # Mark them state 2 now as they are fully explored from the perspective of finding a *new* cycle in this path.
        # This step is necessary to correctly handle paths that merge into already explored parts
        # and ensures the outer loop continues searching in unexplored components.
        # The time array values for these nodes are now irrelevant.
        for node in path:
            state[node] = 2

# The problem guarantees a cycle exists, so the program must find one and call sys.exit().
# Reaching the end of the script should not be possible under valid inputs.