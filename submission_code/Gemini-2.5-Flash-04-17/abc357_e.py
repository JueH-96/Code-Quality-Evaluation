import sys

# Required for larger inputs due to potential graph structure leading to deep paths
# Although the iterative approach mitigates the standard recursion depth limit.
# Keeping this line as it's often useful in competitive programming for graph problems.
# sys.setrecursionlimit(300000)

def solve():
    # Read input N
    N = int(sys.stdin.readline())

    # Read the adjacency list, map 1-based vertex IDs (1..N) to 0-based indices (0..N-1)
    # adj[i] stores the 0-based index of the vertex that vertex i (0-based) points to.
    adj = [0] * N
    a_input = list(map(int, sys.stdin.readline().split()))
    for i in range(N):
        # a_input[i] is the 1-based ID of the neighbor of vertex (i+1)
        # We store the 0-based index (a_input[i] - 1) as the neighbor of vertex i (0-based)
        adj[i] = a_input[i] - 1

    # State array for graph traversal:
    # 0: Unvisited - vertex has not been seen in any traversal yet.
    # 1: Visiting - vertex is currently in the path of the active traversal (on the stack).
    # 2: Visited - vertex has been fully processed as part of a component (cycle or tail).
    state = [0] * N

    # steps_to_cycle[i]: Stores the number of edges from vertex i (0-based)
    # to the first vertex on the cycle that vertex i eventually reaches.
    # If vertex i is on a cycle, steps_to_cycle[i] is 0.
    steps_to_cycle = [-1] * N

    # cycle_id[i]: Stores the ID of the cycle that vertex i (0-based) eventually reaches.
    cycle_id = [-1] * N

    # cycle_sizes: A list storing the size (number of vertices) of each identified cycle.
    # The index in this list is the cycle ID.
    cycle_sizes = []

    # --- Iterative DFS to find components, cycles, and tail lengths ---

    # Iterate through each vertex from 0 to N-1
    for i in range(N):
        # If vertex i has not been visited yet (state 0), start a traversal from it
        if state[i] == 0:
            # List to store the path taken in this traversal (nodes visited in order)
            # This list serves as the implicit stack for the iterative DFS branch.
            path_nodes_in_component = []
            # Dictionary to map vertex to its step count from the start_node of this traversal
            # This helps in calculating distances within the path
            path_step_count = {} # vertex (0-based) -> steps from start_node

            curr = i # Start traversal from vertex i
            steps = 0 # Initialize steps count from start_node

            # Traverse until we hit a node whose state is not Unvisited (0)
            while state[curr] == 0:
                state[curr] = 1 # Mark the current node as Visiting
                path_nodes_in_component.append(curr)
                path_step_count[curr] = steps

                # Move to the next vertex
                curr = adj[curr]
                steps += 1 # Increment steps count

            # After the loop, 'curr' is the vertex we just arrived at (from the last node added to path_nodes_in_component),
            # and its state is either 1 (Visiting - indicates a cycle) or 2 (Visited - indicates hitting an already processed component).

            if state[curr] == 1: # Case 1: Cycle detected! 'curr' is in the current path (state 1)
                # 'curr' is the vertex where the path closes on itself, forming a cycle.
                # It is present in the `path_nodes_in_component` list.
                cycle_start_step_count = path_step_count[curr] # Steps from start_node to the first occurrence of 'curr'
                cycle_length = steps - cycle_start_step_count # The length of the cycle found

                # Assign a new cycle ID and store its size
                k = len(cycle_sizes) # Use the current number of cycles as the new cycle ID
                cycle_sizes.append(cycle_length)

                # Process all nodes that were part of this traversal branch (`path_nodes_in_component`)
                # These nodes now have their component information determined.
                for node in path_nodes_in_component:
                    state[node] = 2 # Mark node as Visited (processing finished for this component)
                    cycle_id[node] = k # Assign the cycle ID

                    node_step_count = path_step_count[node] # Steps from start_node to this node

                    if node_step_count >= cycle_start_step_count:
                        # This node is part of the detected cycle (it occurs at or after the cycle starts in the path)
                        steps_to_cycle[node] = 0 # Nodes on the cycle have 0 steps to reach the cycle

                    else:
                        # This node is in the tail leading to the cycle
                        # The number of edges from this tail node to the cycle entry point (`curr`)
                        # is the difference in steps from the start_node.
                        steps_to_cycle[node] = cycle_start_step_count - node_step_count

                # Component fully processed.

            elif state[curr] == 2: # Case 2: Hit a processed component rooted at 'curr' (state 2)
                # The path from the `start_node` (i) up to the node before 'curr' forms a tail
                # leading into a component that has already been fully processed.
                # All nodes visited in this traversal branch (`path_nodes_in_component`) are in this tail.

                # Get the cycle information from the already processed node 'curr'
                k = cycle_id[curr] # Cycle ID that 'curr' leads to
                steps_from_curr_to_cycle = steps_to_cycle[curr] # Steps from 'curr' to its cycle

                # Process all nodes visited in this component traversal (`path_nodes_in_component`)
                # All these nodes are tail nodes for this component.
                for node in path_nodes_in_component:
                    state[node] = 2 # Mark node as Visited
                    cycle_id[node] = k # Assign the cycle ID

                    node_step_count = path_step_count[node] # Steps from start_node to this node

                    # Calculate the number of edges from 'node' to 'curr'.
                    # The total steps from start_node to 'curr' is `steps`.
                    # The steps from start_node to 'node' is `node_step_count`.
                    # The steps from 'node' to 'curr' is `steps - node_step_count`.
                    steps_from_node_to_curr = steps - node_step_count

                    # Total steps from 'node' to the cycle = (steps from node to curr) + (steps from curr to cycle)
                    steps_to_cycle[node] = steps_from_node_to_curr + steps_from_curr_to_cycle

            # Regardless of the outcome (cycle or hit visited), all nodes in `path_nodes_in_component`
            # have been processed and their state is now 2. The traversal from `i` is complete.

    # --- Calculate the total number of reachable pairs (u, v) ---
    # For a vertex i (0-based), the number of vertices reachable from it is the sum of:
    # 1. The number of vertices in the tail starting from i (including i itself, excluding the first cycle vertex).
    #    This number is exactly `steps_to_cycle[i]`. (Based on logic derived from sample outputs).
    # 2. The number of vertices in the cycle that i eventually reaches.
    #    This number is `cycle_sizes[cycle_id[i]]`.

    total_reachable_pairs = 0
    # Iterate through each vertex from 0 to N-1
    for i in range(N):
        k = cycle_id[i] # Get the cycle ID for vertex i
        # The number of reachable vertices from i is steps_to_cycle[i] + cycle_sizes[k]
        reachable_count = steps_to_cycle[i] + cycle_sizes[k]
        total_reachable_pairs += reachable_count # Add to the total sum

    # Print the final result
    print(total_reachable_pairs)

# Execute the solve function
solve()