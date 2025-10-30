import sys

def solve():
    # Read N, the number of vertices.
    N = int(sys.stdin.readline())

    # Read the A_i values. A_i is 1-indexed, representing the vertex that i points to.
    A_values_1_indexed = list(map(int, sys.stdin.readline().split()))
    
    # Create an adjacency list (array) for 0-indexed vertices.
    # adj[i] will store the 0-indexed vertex that 0-indexed vertex i points to.
    adj = [0] * N
    for i in range(N):
        # Convert 1-indexed A_values[i] to 0-indexed for adj array.
        adj[i] = A_values_1_indexed[i] - 1

    # State array to keep track of vertex visitation status during DFS-like traversal:
    # 0: Unvisited - The vertex has not been encountered yet.
    # 1: Visiting - The vertex is currently in the path being explored (on the current DFS stack).
    # 2: Visited/Processed - The vertex has been fully explored, and its outgoing path (and any cycles
    #                         reachable from it) has been processed.
    state = [0] * N

    # Iterate through all vertices. Since the graph might be disconnected,
    # we need to initiate a traversal from any unvisited vertex.
    for start_node_0_indexed in range(N):
        if state[start_node_0_indexed] == 0:
            # If the current start_node is unvisited, begin a new path traversal.
            
            # This list stores the 0-indexed nodes in the current path being explored.
            current_path_nodes = [] 
            curr_node = start_node_0_indexed
            
            # Follow the path until we encounter a node that is not unvisited (state != 0).
            while state[curr_node] == 0:
                state[curr_node] = 1 # Mark the current node as 'visiting'.
                current_path_nodes.append(curr_node) # Add it to the current path.
                curr_node = adj[curr_node] # Move to the next node in the sequence.
            
            # After the loop, 'curr_node' is the first node encountered that has a state other than '0'.
            # Now, check its specific state to determine what happened:
            
            if state[curr_node] == 1:
                # If state[curr_node] is 1, it means 'curr_node' is already present in our
                # 'current_path_nodes'. This signifies that we have found a cycle.
                # The cycle starts from the first occurrence of 'curr_node' in 'current_path_nodes'
                # and extends to the end of 'current_path_nodes'.
                
                # Find the index of the first occurrence of 'curr_node' in the current path.
                cycle_start_index_in_path = current_path_nodes.index(curr_node)
                
                # Extract the nodes that form the cycle (these are 0-indexed).
                cycle_nodes_0_indexed = current_path_nodes[cycle_start_index_in_path:]
                
                # Convert the cycle nodes to 1-indexed for the required output format.
                cycle_nodes_1_indexed = [node + 1 for node in cycle_nodes_0_indexed]
                
                # Print the length of the cycle.
                print(len(cycle_nodes_1_indexed))
                # Print the cycle vertices, space-separated.
                print(*(cycle_nodes_1_indexed))
                
                # The problem asks for *any* directed cycle, so we can terminate after finding one.
                return
            
            # If state[curr_node] == 2:
            # This means the current path led into a component that has already been fully processed
            # (e.g., a tree leading to a cycle already found, or a part of another tree).
            # The current path itself does not contain a new cycle starting from 'start_node_0_indexed'.
            # Therefore, all nodes in 'current_path_nodes' can now be marked as fully processed (state 2)
            # to prevent re-exploration in subsequent iterations.
            for node_in_path in current_path_nodes:
                state[node_in_path] = 2

# Call the solve function to execute the program logic.
solve()