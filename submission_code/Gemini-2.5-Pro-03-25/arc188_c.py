# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep DSU trees potentially
# The number of nodes is 2*N. N can be up to 2e5, so 2N = 4e5.
# Default recursion limit in Python is often 1000, which might be insufficient.
# Set a sufficiently large limit. Add some buffer. 500000 should be safe.
try:
    sys.setrecursionlimit(500000) 
except Exception as e:
    # Some competitive programming environments might restrict changing the recursion limit.
    # If this fails, the code might fail with RecursionError on large test cases.
    # An iterative implementation of find would be safer in restricted environments.
    # We proceed with the recursive version assuming the limit can be increased.
    # print(f"Warning: Could not set recursion depth limit: {e}", file=sys.stderr) # Optional warning
    pass # Silently ignore if cannot set limit, hope for the best.


def solve():
    N, M = map(int, sys.stdin.readline().split())

    # Initialize Disjoint Set Union (DSU) structure for 2N nodes.
    # Nodes 0 to N-1 represent variables x_1 to x_N.
    #   x_i = 0 if villager i+1 is honest, x_i = 1 if liar.
    # Nodes N to 2N-1 represent variables z_1 to z_N.
    #   z_i = 0 if villager i+1 tells the truth, z_i = 1 if lies.
    # The relationship z_i = x_i ^ y_i connects to the confusion status y_i.
    #   y_i = 1 if villager i+1 is confused, y_i = 0 otherwise.
    
    parent = list(range(2 * N)) # parent[k] stores the parent of node k
    # potential[k] stores the XOR sum relative to the parent:
    # potential[k] = value[k] ^ value[parent[k]]
    potential = [0] * (2 * N) 

    def find(k):
        """
        Finds the root of the component containing node k.
        Also computes the potential of k relative to the root (value[k] ^ value[root]).
        Uses path compression to flatten the tree structure for efficiency.
        """
        if parent[k] == k:
            # Node k is the root. Its potential relative to itself is 0.
            return k, 0
        
        # Recursively find the root of the parent node.
        root, potential_parent_to_root = find(parent[k])
        
        # Update the potential of k.
        # Initially, potential[k] is value[k] ^ value[parent[k]].
        # We need value[k] ^ value[root].
        # This equals (value[k] ^ value[parent[k]]) ^ (value[parent[k]] ^ value[root]).
        # Which is potential[k]_old ^ potential_parent_to_root.
        potential[k] ^= potential_parent_to_root
        
        # Path compression: Make the root the direct parent of k.
        parent[k] = root
        
        # Return the root and the calculated potential of k relative to the root.
        return root, potential[k]

    def unite(u, v, K):
        """
        Processes a constraint of the form value[u] ^ value[v] = K.
        If u and v are in different components, it merges them and sets the relative potential.
        If u and v are already in the same component, it checks if the constraint is consistent
        with the existing structure.
        Returns True if the operation is successful and consistent, False if a contradiction is found.
        """
        # Find the roots and potentials relative to roots for nodes u and v.
        root_u, p_u = find(u) # p_u = value[u] ^ value[root_u]
        root_v, p_v = find(v) # p_v = value[v] ^ value[root_v]
        
        if root_u != root_v:
            # u and v are in different components. Merge them.
            # We merge the component of root_v into the component of root_u.
            parent[root_v] = root_u
            
            # Calculate the potential of root_v relative to its new parent root_u.
            # This required potential is p_rootv_rootu = value[root_v] ^ value[root_u].
            # We use the constraint value[u] ^ value[v] = K.
            # Substitute value[u] = value[root_u] ^ p_u and value[v] = value[root_v] ^ p_v:
            # (value[root_u] ^ p_u) ^ (value[root_v] ^ p_v) = K
            # Rearrange the equation to isolate value[root_v] ^ value[root_u]:
            # value[root_v] ^ value[root_u] = p_u ^ p_v ^ K
            potential[root_v] = p_u ^ p_v ^ K
            return True
        else:
            # u and v are already in the same component.
            # Check if the new constraint K is consistent with the existing potentials.
            # The constraint value[u] ^ value[v] = K must hold.
            # Substitute using potentials relative to the common root (root_u):
            # value[u] = value[root_u] ^ p_u
            # value[v] = value[root_u] ^ p_v
            # (value[root_u] ^ p_u) ^ (value[root_u] ^ p_v) = K
            # Simplify the equation: p_u ^ p_v = K
            if (p_u ^ p_v) == K:
                # The constraint is consistent with the existing structure.
                return True 
            else:
                # The constraint contradicts the existing structure.
                # This indicates an inconsistency in the testimonies given the assumed confusion set.
                # But here, we are solving for existence of ANY valid confusion set.
                # If this check fails, it means the *entire system* including x and z variables is inconsistent.
                return False 

    # Read all testimonies into memory first.
    # This avoids potential issues with standard input buffering if we break early.
    testimonies = []
    for _ in range(M):
        testimonies.append(list(map(int, sys.stdin.readline().split())))

    # Process each testimony and update the DSU structure.
    consistent = True
    for i in range(M):
        A, B, C = testimonies[i] # A, B are 1-based villager indices. C is 0 or 1.
        
        # Convert 1-based villager indices to 0-based array indices.
        A_idx = A - 1 
        B_idx = B - 1
        
        # The problem statement rules translate to the constraint: x_B ^ z_A = C
        # Map these variables to nodes in our DSU structure:
        # x_B corresponds to node index B_idx
        # z_A corresponds to node index N + A_idx
        u_node = B_idx      # Node representing x_B
        v_node = N + A_idx  # Node representing z_A
        
        # Apply the constraint using the unite function.
        # If unite returns False, a contradiction is detected.
        if not unite(u_node, v_node, C):
            consistent = False
            break # Inconsistency found, no solution exists. Stop processing.

    # After processing all testimonies, check if the system was consistent.
    if not consistent:
        # A contradiction was found, meaning no assignment of honest/liar statuses 
        # and truth-telling statuses can satisfy all testimonies simultaneously.
        print("-1")
    else:
        # The system is consistent. There exists at least one valid assignment for x_i and z_i.
        # We need to find one such assignment to compute the confusion statuses y_i.
        # We can determine values by assigning a value (e.g., 0) to the root of each component
        # and propagating values using the computed potentials.
        
        final_values = [-1] * (2 * N) # Stores the computed values for x_i and z_i variables. -1 indicates unassigned.
        
        for k in range(2 * N): # Iterate through all nodes (variables)
            if final_values[k] == -1: # If the value for this node hasn't been determined yet
                root, p_k = find(k) # Find the root and potential relative to root (p_k = value[k] ^ value[root])
                
                # If the root's value is also unassigned, assign it a default value of 0.
                # This choice determines which specific solution we find among potentially many.
                if final_values[root] == -1:
                     final_values[root] = 0
                
                # Calculate the value for node k using the root's value and the relative potential.
                # value[k] = value[root] ^ p_k
                final_values[k] = final_values[root] ^ p_k

        # Now that all x_i and z_i values are computed in final_values array,
        # compute the confusion status y_i for each villager i.
        # The relationship is y_i = x_i ^ z_i.
        y_result = [] # List to build the output string
        for i in range(N): # Iterate through villagers 1 to N (represented by indices 0 to N-1)
            
            # Retrieve the computed values for x_{i+1} and z_{i+1}
            x_i_val = final_values[i]       # value[x_{i+1}] is stored at index i
            z_i_val = final_values[N + i]   # value[z_{i+1}] is stored at index N+i
            
            # Compute the confusion status y_{i+1} = x_{i+1} ^ z_{i+1}
            y_i = x_i_val ^ z_i_val # y_i=1 means confused, y_i=0 means not confused.
            y_result.append(str(y_i)) # Append the status as a character '0' or '1'
            
        # Print the final result string which represents one valid set of confused villagers.
        print("".join(y_result))

# Execute the main solution logic.
solve()