import sys

# Potentially increase recursion depth for DSU find operations with path compression.
# The theoretical amortized complexity holds without increasing recursion depth,
# but Python's default limit might be hit by deep recursion stacks in worst-case
# scenarios before path compression significantly flattens the tree.
# For N up to 2e5, a very deep recursion stack *could* occur in a worst-case
# path compression scenario before the tree becomes flat, exceeding the default
# limit (often 1000). However, typical competitive programming platforms
# might have a higher limit or the test cases are not worst-case for this.
# Let's rely on the default for now. If encountering RecursionError, uncomment below.
# sys.setrecursionlimit(max(sys.getrecursionlimit(), 200005))


def find(j, par, diff):
    """
    Finds the root of element j and the XOR difference between A[j] and A[root].
    Performs path compression.
    Returns (root, diff_from_j_to_root), where A[j] ^ A[root] = diff_from_j_to_root.
    Note: This operates on the bit value a_{j,k} for the current bit k.
    """
    if par[j] == j:
        return (j, 0)
    
    # Recursively find the root and difference for the parent
    root, diff_from_parent_to_root = find(par[j], par, diff)
    
    # Update the difference for j during path compression
    # We know:
    # a_{j,k} ^ a_{par[j], k} = diff[j] (value stored when parent was par[j])
    # a_{par[j], k} ^ a_{root, k} = diff_from_parent_to_root (result from recursive call)
    # XORing these gives:
    # (a_{j,k} ^ a_{par[j], k}) ^ (a_{par[j], k} ^ a_{root, k}) = diff[j] ^ diff_from_parent_to_root
    # a_{j,k} ^ a_{root, k} = diff[j] ^ diff_from_parent_to_root
    # Update diff[j] to store the difference from j to the root
    diff[j] = diff[j] ^ diff_from_parent_to_root
    
    # Update the parent of j to the root
    par[j] = root
    
    return (root, diff[j])

def union(u, v, c, par, diff):
    """
    Unites the sets containing u and v based on the constraint a_{u,k} ^ a_{v,k} = c.
    Returns True if successful or consistent, False if inconsistency detected.
    """
    root_u, diff_u = find(u, par, diff)
    root_v, diff_v = find(v, par, diff)

    if root_u == root_v:
        # If u and v are already in the same set, check for consistency.
        # From find results: a_{u,k} ^ a_{root_u, k} = diff_u and a_{v,k} ^ a_{root_v, k} = diff_v.
        # Since root_u == root_v, we have a_{u,k} ^ a_{root_u, k} = diff_u and a_{v,k} ^ a_{root_u, k} = diff_v.
        # XORing these gives (a_{u,k} ^ a_{root_u, k}) ^ (a_{v,k} ^ a_{root_u, k}) = diff_u ^ diff_v.
        # This simplifies to a_{u,k} ^ a_{v,k} = diff_u ^ diff_v.
        # The constraint is a_{u,k} ^ a_{v,k} = c.
        # Consistency requires c == diff_u ^ diff_v.
        if (diff_u ^ diff_v) != c:
            return False # Inconsistency detected for this bit
        return True # Constraint is consistent with existing structure

    # If u and v are in different sets, merge them. Merge root_v's set into root_u's set.
    par[root_v] = root_u
    
    # We need to set diff[root_v] such that a_{root_v, k} ^ a_{par[root_v], k} = diff[root_v],
    # which is a_{root_v, k} ^ a_{root_u, k} = diff[root_v].
    # We know:
    # a_{u,k} ^ a_{v,k} = c (constraint)
    # a_{u,k} = a_{root_u, k} ^ diff_u (from find(u))
    # a_{v,k} = a_{root_v, k} ^ diff_v (from find(v))
    # Substitute into the constraint:
    # (a_{root_u, k} ^ diff_u) ^ (a_{root_v, k} ^ diff_v) = c
    # a_{root_u, k} ^ a_{root_v, k} = c ^ diff_u ^ diff_v
    # We want diff[root_v] = a_{root_v, k} ^ a_{root_u, k}.
    # a_{root_v, k} ^ a_{root_u, k} = (a_{root_u, k} ^ a_{root_v, k}) ^ 0 = (c ^ diff_u ^ diff_v) ^ 0 = c ^ diff_u ^ diff_v.
    # So, set diff[root_v] to c ^ diff_u ^ diff_v.
    diff[root_v] = diff_u ^ diff_v ^ c
    
    return True

# Read input
N, M = map(int, sys.stdin.readline().split())
constraints = []
for _ in range(M):
    x, y, z = map(int, sys.stdin.readline().split())
    constraints.append((x, y, z))

# Initialize result array for A_1 to A_N (using 1-based indexing)
A = [0] * (N + 1)

# Iterate through bits from 0 upwards.
# The maximum Z_i is 10^9. 2^29 approx 5.3e8, 2^30 approx 1.07e9.
# The highest possible set bit in any Z_i <= 10^9 is bit 29 (value 2^29).
# Bits 0 through 29 are relevant. For bits k >= 30, Z_i's k-th bit is always 0.
# Constraint a_{X_i, k} ^ a_{Y_i, k} = 0 means a_{X_i, k} = a_{Y_i, k}.
# For such bits, all elements within a connected component must have the same bit value.
# To minimize sum, we set this common bit value to 0 for all components.
# So, we only need to consider bits k from 0 to 29.
max_bits = 30 

possible = True
for k in range(max_bits):
    # Initialize DSU structure for the current bit k
    par = list(range(N + 1))
    diff = [0] * (N + 1)
    
    current_bit_possible = True
    
    # Apply constraints for the current bit k
    for x, y, z in constraints:
        z_k = (z >> k) & 1 # Get the k-th bit of Z_i
        if not union(x, y, z_k, par, diff):
            # If union results in an inconsistency, a good sequence does not exist.
            current_bit_possible = False
            break # Stop processing constraints for this bit

    if not current_bit_possible:
        possible = False
        break # No solution exists globally, stop processing further bits

    # If current bit is possible, determine the optimal bit values for each element
    
    # Gather information about components and differences relative to their roots
    # We use a dictionary to store counts for each root: {root: [count_of_diff_0, count_of_diff_1]}
    root_info = {} 

    # Iterate through all elements (1 to N) to find their roots and differences relative to the root.
    # This pass also helps complete the path compression for the current bit k, making subsequent finds faster.
    for j in range(1, N + 1):
        # Note: Calling find again is important because path compression might have updated the tree structure.
        root_j, diff_j = find(j, par, diff) 
        if root_j not in root_info:
            root_info[root_j] = [0, 0]
        
        # diff_j represents a_{j,k} ^ a_{root_j, k}.
        # If we set a_{root_j, k} = 0, then a_{j,k} = diff_j.
        # If we set a_{root_j, k} = 1, then a_{j,k} = 1 ^ diff_j.
        # The count_0 entry stores the number of elements j in the component where diff_j = 0.
        # If a_{root_j, k} = 1, these elements j will have a_{j,k} = 1 ^ 0 = 1.
        # The count_1 entry stores the number of elements j in the component where diff_j = 1.
        # If a_{root_j, k} = 0, these elements j will have a_{j,k} = 0 ^ 1 = 1.
        root_info[root_j][diff_j] += 1

    # Determine the optimal bit value (0 or 1) for the root of each component
    # to minimize the sum of bits (sum of a_{j,k}) for this component at this bit position.
    optimal_root_bits = {} # {root: optimal_a_root_k}
    for root in root_info:
        # Roots that were not touched by any constraint will not be in root_info,
        # or if they are, their component only contains themselves.
        # Elements j that are their own root (par[j]==j) and were not involved in any union call
        # will have root_j = j, diff_j = 0. root_info[j] will be [1, 0].
        # count_0 = 1, count_1 = 0. optimal_root_bits[j] = 0 (since 0 <= 1).
        # This correctly assigns bit 0 to isolated nodes, minimizing sum.
        
        # Handle potential roots that are keys in root_info but might not have been the root of any union.
        # This case is covered by the loop `for root in root_info:`.
        
        count_0, count_1 = root_info[root]
        
        # If we set a_{root, k} = 0, the number of 1s in the component is count_1.
        # If we set a_{root, k} = 1, the number of 1s in the component is count_0.
        # To minimize the sum of bits for this component, choose a_{root, k} = 0 if count_1 <= count_0,
        # otherwise choose a_{root, k} = 1.
        optimal_root_bits[root] = 0 if count_1 <= count_0 else 1

    # Assign the determined bit value a_{j,k} for each element j in the component
    # and add its contribution (a_{j,k} * 2^k) to the total value A[j].
    for j in range(1, N + 1):
        # Find the root and difference again. This second find pass per bit is crucial
        # as the first pass might not have fully compressed all paths yet.
        # It ensures we use the final DSU structure for this bit.
        root_j, diff_j = find(j, par, diff)
        
        # The bit value for element j is a_{j,k} = a_{root_j, k} ^ (a_{j,k} ^ a_{root_j, k})
        # This is the optimal root bit XORed with the difference from j to root.
        # If root_j was not in root_info (meaning j was an isolated node not touched by any constraint),
        # root_j == j, diff_j == 0. In this case, optimal_root_bits[j] should be 0.
        # The loop `for j in range(1, N + 1):` before this iterates over all nodes,
        # calling find(j) which adds `j` to `root_info` if it becomes a root, or finds its root.
        # If `j` remains its own root and is not involved in any constraint, `find(j)` returns `(j, 0)`.
        # `root_info[j]` becomes `[1, 0]`. `optimal_root_bits[j]` becomes 0. The logic holds.
        a_j_k = optimal_root_bits[root_j] ^ diff_j
        
        # Add the value 2^k if the k-th bit of A[j] is 1
        A[j] += a_j_k * (1 << k)

# After iterating through all relevant bits (0 to 29)
if not possible:
    # If any bit was impossible to satisfy the constraints, no good sequence exists.
    print(-1)
else:
    # If all bits were possible, print the resulting sequence A_1 to A_N, separated by spaces.
    # A is 1-indexed, so print from A[1] onwards.
    print(*A[1:])