# YOUR CODE HERE
import sys

# Function for modular exponentiation - Python's pow does this efficiently.
# pow(base, exponent, modulus) calculates (base ** exponent) % modulus

def solve():
    # Read N and K from standard input
    N, K = map(int, sys.stdin.readline().split())
    
    # Read the permutation P (1-based indexing)
    P = list(map(int, sys.stdin.readline().split()))

    # Convert P to 0-based indexing for easier array manipulation.
    # P_0idx[i] stores the element that element i maps to (0-based).
    # Specifically, if the input permutation is P = (P_1, P_2, ..., P_N),
    # then P_0idx[i] = P_{i+1} - 1.
    # This array represents the initial permutation function f_0, where f_0(i) = P_0idx[i].
    P_0idx = [p - 1 for p in P]

    # final_P_0idx will store the resulting permutation P^(K) in 0-based indexing.
    # After computation, final_P_0idx[i] will hold the value P^{(K)}_{i+1} - 1.
    final_P_0idx = [0] * N
    
    # visited array to keep track of elements already assigned to a cycle and processed.
    visited = [False] * N

    # Iterate through all elements from 0 to N-1 to find cycles.
    for i in range(N):
        # If element i has already been visited (part of a previously found cycle), skip it.
        if not visited[i]:
            
            # Found an unvisited element, start tracing its cycle.
            current_cycle = [] # List to store elements of the cycle in the order they appear.
            curr = i # Start tracing from element i.
            
            # Trace the cycle using the P_0idx mapping.
            # Keep moving to the next element in the sequence i -> P_0idx[i] -> P_0idx[P_0idx[i]] -> ...
            # Stop when an element marked as visited is encountered. Since P is a permutation,
            # this process will eventually return to the starting element i, completing the cycle.
            while not visited[curr]:
                visited[curr] = True # Mark the current element as visited.
                current_cycle.append(curr) # Add the element to the cycle list.
                # Move to the next element in the cycle as defined by the permutation P_0idx.
                curr = P_0idx[curr] 
            
            # Calculate the length of the found cycle.
            cycle_len = len(current_cycle)
            
            # Basic check: If cycle_len is 0, something is wrong (should not happen for N >= 1).
            if cycle_len == 0:
                 continue 

            # The problem describes an operation that, when applied K times, results in a permutation
            # corresponding to applying the initial permutation function f_0 exactly 2^K times.
            # That is, the final state P^(K) is given by P^(K)_i = f_0^{2^K}(i).
            # To compute f_0^{2^K}(i) for an element i within a cycle of length L,
            # we only need to consider the exponent modulo L due to the cyclic nature.
            # So we need to compute f_0^{E}(i) where E = 2^K mod L.
            # This is equivalent to moving E steps along the cycle starting from i.
            
            # Calculate E = 2^K mod cycle_len using Python's built-in pow function,
            # which handles modular exponentiation efficiently, even for large K.
            steps_to_take = pow(2, K, cycle_len)

            # Now, determine the final position for each element in the current cycle.
            for j in range(cycle_len):
                # start_node is the element at index j in the `current_cycle` list.
                start_node = current_cycle[j]
                
                # To find where start_node ends up after E steps, we find its new index in the cycle list.
                # The new index is (j + steps_to_take) mod cycle_len.
                final_node_idx_in_cycle = (j + steps_to_take) % cycle_len
                
                # The element value at this final index in the cycle list is final_node.
                final_node = current_cycle[final_node_idx_in_cycle]
                
                # In the final permutation P^(K), the element at index `start_node` becomes `final_node`.
                # We store this result in our 0-based final permutation array.
                final_P_0idx[start_node] = final_node

    # Convert the final 0-based permutation array back to 1-based indexing for output.
    # If final_P_0idx[i] = val, this means P^(K)_{i+1} - 1 = val, so P^(K)_{i+1} = val + 1.
    final_P_1idx = [val + 1 for val in final_P_0idx]
    
    # Print the resulting permutation elements separated by spaces.
    print(*(final_P_1idx))

# Call the solve function to execute the solution logic when the script is run.
solve()

# END OF YOUR CODE HERE