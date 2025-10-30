# YOUR CODE HERE
import sys
from collections import defaultdict

# Read N and M
N, M = map(int, sys.stdin.readline().split())

# DSU structures
# parent[i] stores the parent of element i
# xor_from_root[i] stores the XOR sum of values from A_i to A_parent[i].
# After path compression in find(i), xor_from_root[i] stores A_i ^ A_{root_of_i}
parent = list(range(N + 1))
xor_from_root = [0] * (N + 1)

# Function to find the root of an element i and the XOR sum from i to its root
def find(i):
    if parent[i] == i:
        return i, xor_from_root[i]
    
    # Path compression: make i's parent the root
    # Update xor_from_root[i] to be A_i ^ A_root
    root, xor_val_to_parent_root = find(parent[i])
    # xor_from_root[i] currently stores (A_i ^ A_old_parent_of_i)
    # xor_val_to_parent_root stores (A_old_parent_of_i ^ A_root)
    # So, (A_i ^ A_old_parent_of_i) ^ (A_old_parent_of_i ^ A_root) = (A_i ^ A_root)
    xor_from_root[i] = xor_from_root[i] ^ xor_val_to_parent_root
    parent[i] = root
    return root, xor_from_root[i]

# Function to unite two sets based on the XOR condition A_u ^ A_v = z
# Returns True if successful, False if a contradiction is found
def union(u, v, z):
    root_u, xor_u_to_root_u = find(u)
    root_v, xor_v_to_root_v = find(v)

    if root_u != root_v:
        # Link root_u to root_v (arbitrary choice, root_v becomes new root)
        parent[root_u] = root_v
        # Calculate xor_from_root[root_u] which represents A_root_u ^ A_root_v
        # We have:
        # A_u = A_root_u ^ xor_u_to_root_u
        # A_v = A_root_v ^ xor_v_to_root_v
        # Given A_u ^ A_v = z
        # Substitute: (A_root_u ^ xor_u_to_root_u) ^ (A_root_v ^ xor_v_to_root_v) = z
        # Rearrange: A_root_u ^ A_root_v = z ^ xor_u_to_root_u ^ xor_v_to_root_v
        xor_from_root[root_u] = z ^ xor_u_to_root_u ^ xor_v_to_root_v
        return True
    else:
        # u and v are already in the same component
        # Check for consistency: the implied A_u ^ A_v must equal z
        # Implied A_u ^ A_v = (A_root_u ^ xor_u_to_root_u) ^ (A_root_v ^ xor_v_to_root_v)
        # Since root_u == root_v, this simplifies to xor_u_to_root_u ^ xor_v_to_root_v
        current_xor_uv = xor_u_to_root_u ^ xor_v_to_root_v
        if current_xor_uv != z:
            return False # Contradiction
        return True

# Process all M conditions
possible = True
for _ in range(M):
    X_i, Y_i, Z_i = map(int, sys.stdin.readline().split())
    if not union(X_i, Y_i, Z_i):
        possible = False
        break

if not possible:
    print(-1)
else:
    # After all unions, ensure all nodes have their parent pointing directly to the root
    # and xor_from_root stores the correct A_i ^ A_root value.
    # Also collect all relative XOR values for each component.
    xor_vals_in_component = defaultdict(list) # Key: root node, Value: list of (A_i ^ A_root) for nodes in component
    for i in range(1, N + 1):
        root, xor_val_to_root = find(i) # This call performs path compression and updates xor_from_root[i]
        xor_vals_in_component[root].append(xor_val_to_root)
    
    # Determine the optimal A_i values
    A = [0] * (N + 1) # 1-indexed to match problem statement

    # Z_i values are up to 10^9, which is less than 2^30.
    # So we need to consider bits from 0 up to 29 (30 bits total).
    MAX_BITS = 30 

    for root_node, relative_xor_values in xor_vals_in_component.items():
        optimal_K_for_component = 0 # This will be the assigned value for A_root_node
        
        # Determine each bit of optimal_K_for_component
        for b in range(MAX_BITS):
            count_bit_0_in_relative_xor = 0
            count_bit_1_in_relative_xor = 0
            
            for val in relative_xor_values:
                if (val >> b) & 1:
                    count_bit_1_in_relative_xor += 1
                else:
                    count_bit_0_in_relative_xor += 1
            
            # To minimize the sum of A_i's for this component, we choose each bit K_b
            # such that sum_{i in component} (K_b ^ (relative_xor_values[i])_b) is minimized.
            # If K_b = 0, sum contrib = count_bit_1_in_relative_xor (count of 1s)
            # If K_b = 1, sum contrib = count_bit_0_in_relative_xor (count of 0s)
            # We want to pick the K_b that results in the smaller count.
            if count_bit_0_in_relative_xor < count_bit_1_in_relative_xor:
                # If we set K_b = 1, it flips 0s to 1s. This is better if count of 0s is smaller.
                optimal_K_for_component |= (1 << b)
            # else (count_bit_1_in_relative_xor <= count_bit_0_in_relative_xor),
            # setting K_b = 0 is better or equal. So leave the bit as 0.

        # Assign actual A_i values for all nodes in this component
        # Because we called find(i) for all i from 1 to N earlier,
        # parent[i] now points directly to its root and xor_from_root[i] stores A_i ^ A_root.
        for i in range(1, N + 1):
            if parent[i] == root_node: # Check if 'i' belongs to the current component 'root_node'
                A[i] = optimal_K_for_component ^ xor_from_root[i]
    
    # Print the result (A_1 to A_N, space-separated)
    print(*(A[1:]))