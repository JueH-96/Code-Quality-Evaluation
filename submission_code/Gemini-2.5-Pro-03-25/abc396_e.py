# YOUR CODE HERE
import sys

# Set recursion depth limit to handle potentially deep paths in DSU operations.
# N <= 2e5. The maximum recursion depth needed could theoretically be N in a path graph scenario.
# Set a safe large value, e.g., slightly larger than max N.
# Use try-except block for environments where setting recursion limit might fail (e.g., restricted platforms).
try:
    # Increase recursion depth as DSU path compression on deep structures might exceed default limit
    # N is up to 2*10^5, so set it a bit higher
    sys.setrecursionlimit(2 * (10**5) + 10) 
except Exception: 
    # If setting recursion depth fails (e.g., due to system restrictions), 
    # proceed with the default limit. This might lead to RecursionError on large test cases.
    pass 

def solve():
    # Read N (number of elements in sequence A) and M (number of constraints)
    N, M = map(int, sys.stdin.readline().split())
    
    # Read M constraints (X_i, Y_i, Z_i)
    constraints = []
    for _ in range(M):
        constraints.append(list(map(int, sys.stdin.readline().split())))

    # Initialize the result array A with zeros. We use 1-based indexing for A, so size is N+1.
    A = [0] * (N + 1)
    
    # The maximum value of Z_i is 10^9, which is less than 2^30.
    # So we only need to consider bits from 0 up to 29. MAX_BIT = 30 indicates processing bits 0..29.
    MAX_BIT = 30 

    # DSU (Disjoint Set Union) structure variables:
    # parent[i]: stores the parent of node i. If parent[i] == i, then i is a root.
    # potential[i]: stores the XOR sum difference relative to its parent. Specifically, 
    #               initially stores value[i] ^ value[parent[i]]. After path compression via find, 
    #               it stores value[i] ^ value[root].
    # size[i]: stores the size of the component if i is a root. Used for union-by-size heuristic.
    # count1[i]: stores the count of nodes `j` in the component rooted at `i` such that 
    #              value[j] == 1, assuming value[i] == 0. Used for minimizing sum.
    # These arrays will be re-initialized for each bit position k.
    parent = list(range(N + 1))
    potential = [0] * (N + 1) 
    size = [1] * (N + 1)
    count1 = [0] * (N + 1) 
    
    # Recursive find function with path compression and potential update for XOR values
    def find_recursive(i):
      # Base case: node is its own parent (it's a root)
      if parent[i] == i:
          return i, 0 # Return root and potential 0 relative to itself
      
      # Recursive call to find the root of the parent
      root, parent_potential = find_recursive(parent[i])
      
      # Update potential of current node `i` relative to the root
      # The new potential (value[i] ^ value[root]) is potential[i] ^ parent_potential
      # Here potential[i] is value[i] ^ value[parent[i]] before path compression updates it
      # parent_potential is value[parent[i]] ^ value[root] returned by recursive call
      potential[i] = potential[i] ^ parent_potential
      
      # Path compression: make `i` point directly to the root
      parent[i] = root
      
      # Return the root and the updated potential of i relative to the root
      return root, potential[i]

    # Unite function using recursive find. Performs union of sets representing connected components 
    # and checks consistency of constraints.
    def unite(u, v, z_k):
        # Find roots and potentials relative to roots for u and v
        root_u, pot_u = find_recursive(u) 
        root_v, pot_v = find_recursive(v) 
        
        if root_u != root_v:
            # If u and v are in different components, merge them
            # Use union by size heuristic: attach smaller tree to root of larger tree
            # This helps keep the tree depth low, improving performance.
            if size[root_u] < size[root_v]:
                # Swap roles to ensure root_u is the root of the larger (or equal size) component
                root_u, root_v = root_v, root_u 
                pot_u, pot_v = pot_v, pot_u # Swap potentials accordingly

            # Make root_u the parent of root_v (attach root_v tree under root_u)
            parent[root_v] = root_u
            
            # Calculate the potential of root_v relative to its new parent root_u
            # We need potential[root_v] = value[root_v] ^ value[parent[root_v]] = value[root_v] ^ value[root_u]
            # The constraint A[u] ^ A[v] = z_k implies value[u] ^ value[v] = z_k for the current bit k.
            # Substituting known potentials relative to their roots: (value[root_u] ^ pot_u) ^ (value[root_v] ^ pot_v) = z_k
            # Rearranging gives the relationship between root values: value[root_u] ^ value[root_v] = z_k ^ pot_u ^ pot_v
            # This value is stored as potential[root_v] because root_u is the new parent
            P = z_k ^ pot_u ^ pot_v
            potential[root_v] = P
            
            # Update component size and count1 data for the new root root_u
            S0_u = count1[root_u] # Current count of 1s in component u if root_u value is 0
            sz_u = size[root_u]   # Current size of component u
            S0_v = count1[root_v] # Current count of 1s in component v if root_v value is 0
            sz_v = size[root_v]   # Current size of component v
            
            # Calculate new count1 for the merged component rooted at root_u.
            # This represents the total count of 1s in the merged component if we assume value[root_u] = 0.
            new_S0 = S0_u # Start with count from component u
            if P == 0: # If root_v has the same value as root_u (when root_u=0), then value[root_v]=0.
                # The count of 1s in component v remains S0_v.
                new_S0 += S0_v 
            else: # P == 1, root_v has the opposite value of root_u (when root_u=0), then value[root_v]=1.
                  # All values in component v are flipped relative to the assignment where root_v=0.
                  # The number of 1s contributed by component v becomes sz_v - S0_v.
                new_S0 += (sz_v - S0_v) 
                
            count1[root_u] = new_S0 # Update count1 for the merged component root
            size[root_u] = sz_u + sz_v # Update size for the merged component root
            
            return True # Merge successful and consistent
        else: # root_u == root_v, u and v are already in the same component
            # Check for consistency with the new constraint
            # The constraint is value[u] ^ value[v] = z_k
            # We know value[u] = value[root_u] ^ pot_u and value[v] = value[root_v] ^ pot_v.
            # Since root_u == root_v, this becomes (value[root] ^ pot_u) ^ (value[root] ^ pot_v) = z_k
            # This simplifies to pot_u ^ pot_v = z_k
            if (pot_u ^ pot_v) == z_k:
                return True # Constraint is consistent with existing information
            else:
                # Constraint contradicts existing information, indicating an inconsistency.
                return False 

    # Iterate through each bit position k from 0 up to MAX_BIT-1 (0 to 29)
    for k in range(MAX_BIT):
        # Re-initialize DSU structures for the current bit k
        parent = list(range(N + 1))
        potential = [0] * (N + 1) 
        size = [1] * (N + 1)
        count1 = [0] * (N + 1) 

        consistent = True
        # Process all M constraints for the current bit k
        for i in range(M):
            X_i, Y_i, Z_i = constraints[i] # Get constraint data
            z_k = (Z_i >> k) & 1 # Extract the k-th bit of Z_i
            # Apply the constraint using DSU unite operation
            if not unite(X_i, Y_i, z_k):
                # If unite returns False, an inconsistency is detected
                consistent = False
                break # Stop processing constraints for this bit
        
        # If any inconsistency was found for this bit, no solution exists for the entire problem
        if not consistent:
            print("-1")
            return # Exit the program

        # If all constraints are consistent for bit k:
        # Finalize component structures by ensuring all nodes point directly to their final root
        # and potentials are updated relative to the root.
        # Determine the optimal assignment for this bit to minimize sum.
        
        root_val = {} # Dictionary to store the chosen value (0 or 1) for each component root
        
        # This pass ensures all nodes are fully path compressed & potentials finalized relative to true root.
        # This step is crucial after all unions to ensure correct parent pointers and potentials for final calculations.
        for j in range(1, N + 1):
            find_recursive(j) 

        # Determine the optimal value (0 or 1) for each component root to minimize the count of 1s for this bit
        for j in range(1, N + 1):
           if parent[j] == j: # Check if j is a root of a component
               # Process each root only once to avoid redundant calculations
               if j not in root_val: 
                   S0 = count1[j] # Number of 1s if root value is 0
                   s = size[j]    # Total size of the component
                   S1 = s - S0    # Number of 1s if root value is 1
                   
                   # Choose the assignment (root value 0 or 1) that results in fewer 1s
                   if S0 <= S1:
                       root_val[j] = 0 # Assign 0 to root to minimize 1s count
                   else:
                       root_val[j] = 1 # Assign 1 to root to minimize 1s count

        # Calculate the final bit value a_jk for each element A[j] based on the optimal assignment
        # and update the integer value A[j] by adding 2^k if the bit is 1.
        for j in range(1, N + 1):
            # parent[j] is the final root of the component containing j after path compression
            root = parent[j] 
            # potential[j] is value[j] ^ value[root] after path compression
            pot_j = potential[j] 
            
            # Determine the actual value of the k-th bit for A[j]
            # value[j] = value[root] ^ potential[j]
            a_jk = root_val[root] ^ pot_j
            
            # If the k-th bit is 1, add its corresponding power of 2 (1 << k) to A[j]
            if a_jk == 1:
                 A[j] += (1 << k)

    # After processing all bits, print the resulting sequence A (elements A[1] through A[N])
    # Use `*` operator to unpack the list elements separated by spaces.
    print(*(A[1:]))

# Call the main solve function to run the program
solve()