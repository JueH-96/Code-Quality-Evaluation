# YOUR CODE HERE
import sys

# Setting a reasonable recursion depth limit for safety, although iterative find should avoid deep recursion.
# sys.setrecursionlimit(2000010) 

MOD = 998244353

# Global lists for DSU structure: parent array and size array
# These are reset for each test case in the solve function.
parent = []
size = []

# Iterative find function with path compression
def find_iterative(i):
    """Finds the root of the set containing element i, with path compression."""
    root = i
    # Traverse up the tree to find the root
    # Keep track of the path for path compression later
    path = []
    while parent[root] != root:
        path.append(root)
        root = parent[root]
    
    # Apply path compression: make all nodes on the path point directly to the root
    for node in path:
        parent[node] = root
    return root

# Unite function for merging sets based on equality constraints (a = b)
# It also handles implicit consistency for negations (neg(a) = neg(b))
# Returns False if a contradiction is detected during merge, True otherwise.
def unite(i, j):
    """Merges the sets containing i and j. Returns False on contradiction."""
    
    # Find the roots of the sets containing literals i, j, and their negations
    root_i = find_iterative(i)
    root_j = find_iterative(j)
    
    # Literal negation is index XOR 1. E.g., neg(2k) = 2k+1, neg(2k+1) = 2k.
    neg_i = i ^ 1
    neg_j = j ^ 1
    root_neg_i = find_iterative(neg_i)
    root_neg_j = find_iterative(neg_j)

    # Check for contradiction: If we are trying to merge i and j (make i = j),
    # but it's already established that i != j (i.e., i is in the same set as neg(j)),
    # then this operation leads to a contradiction.
    if root_i == root_neg_j:  
        return False # Contradiction detected

    # If i and j are already in the same set, no action needed.
    if root_i != root_j:
        # Union by size heuristic: attach the smaller tree to the root of the larger tree
        if size[root_i] < size[root_j]:
            # Swap roots to ensure root_i represents the larger set (or equal size)
            root_i, root_j = root_j, root_i
            # Swap negation roots accordingly
            root_neg_i, root_neg_j = root_neg_j, root_neg_i
        
        # Merge the set of j into the set of i
        parent[root_j] = root_i
        size[root_i] += size[root_j] # Update size of the merged set
        
        # Consistently merge the negation sets: merge neg(j) into neg(i)
        # It is crucial that the negation consistency is maintained.
        parent[root_neg_j] = root_neg_i
        size[root_neg_i] += size[root_neg_j] # Update size of the merged negation set

    # Merge successful (or unnecessary), no contradiction found related to this operation.
    return True

# Modular exponentiation function: computes (base^exp) % MOD efficiently
def power(base, exp):
    """Computes (base^exp) % MOD using binary exponentiation."""
    res = 1
    base %= MOD
    while exp > 0:
        if exp % 2 == 1: # If exponent is odd, multiply current result by base
            res = (res * base) % MOD
        base = (base * base) % MOD # Square the base
        exp //= 2 # Halve the exponent
    return res

# Main solver function for processing each test case
def solve():
    H, W = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(H)]

    # Preliminary check: Count 'A' tiles per row and column.
    # A necessary condition for a valid tiling to exist is that these counts must be even.
    row_A_counts = [0] * H
    col_A_counts = [0] * W
    for r in range(H):
        for c in range(W):
            if S[r][c] == 'A':
                row_A_counts[r] += 1
                col_A_counts[c] += 1

    valid_parity = True
    # Check row parities
    for r in range(H):
        if row_A_counts[r] % 2 != 0:
            valid_parity = False
            break
    # Check column parities only if row parities are okay
    if valid_parity:
        for c in range(W):
            if col_A_counts[c] % 2 != 0:
                valid_parity = False
                break

    # If the parity condition fails, no solution exists. Output 0.
    if not valid_parity:
        print(0)
        return

    # Initialize Disjoint Set Union (DSU) structure
    # Total variables = HW horizontal edge vars + HW vertical edge vars = 2HW
    num_vars = 2 * H * W
    # Each variable 'x' corresponds to two literals: 'x' (positive) and 'neg(x)' (negative).
    # Total literals = 2 * num_vars = 4HW.
    num_literals = 2 * num_vars 
    
    # Declare parent and size arrays as global to be accessible by helper functions
    # These need to be reset for each test case.
    global parent, size
    # Initialize parent array: each literal is its own parent initially.
    parent = list(range(num_literals)) 
    # Initialize size array: each set initially contains 1 element.
    size = [1] * num_literals

    # Variable index mapping convention:
    # Horizontal edge var `u_rc` (edge right of cell(r,c)): index `r*W + c`. Range `0` to `HW-1`.
    # Vertical edge var `v_rc` (edge below cell(r,c)): index `HW + r*W + c`. Range `HW` to `2HW-1`.
    
    # Process constraints imposed by each cell's tile type
    for r in range(H):
        for c in range(W):
            
            # Calculate variable indices for the four edges adjacent to cell (r, c)
            # Using modulo arithmetic to handle torus wraparound.
            
            # Top edge corresponds to vertical variable below cell ((r-1+H)%H, c)
            v_above_var_idx = H*W + ((r - 1 + H) % H) * W + c
            
            # Bottom edge corresponds to vertical variable below cell (r, c)
            v_below_var_idx = H*W + r*W + c
            
            # Left edge corresponds to horizontal variable right of cell (r, (c-1+W)%W)
            u_left_var_idx = r*W + ((c - 1 + W) % W)

            # Right edge corresponds to horizontal variable right of cell (r, c)
            u_right_var_idx = r*W + c

            # Convert variable indices to literal indices for DSU operations
            # Positive literal for variable index `idx` is `2*idx`.
            # Negative literal for variable index `idx` is `2*idx + 1`.
            v_above = 2 * v_above_var_idx
            v_below = 2 * v_below_var_idx
            u_left = 2 * u_left_var_idx
            u_right = 2 * u_right_var_idx
            
            # Apply constraints based on the tile type S[r][c]
            if S[r][c] == 'A':
                # Type 'A' tile constraints: connect adjacent edges => edge states must differ.
                # Constraint: v_above != v_below. This implies `unite(v_above, neg(v_below))`.
                # Use `a ^ 1` to get the index of the negation literal `neg(a)`.
                # Note: unite(a, b^1) handles the logic for a != b constraint.
                if not unite(v_above, v_below ^ 1): 
                    print(0) # Contradiction detected
                    return

                # Constraint: u_left != u_right. This implies `unite(u_left, neg(u_right))`.
                if not unite(u_left, u_right ^ 1):
                    print(0) # Contradiction detected
                    return

            else: # S[r][c] == 'B'
                # Type 'B' tile constraints: connect opposite edges => edge states must match for pairs.
                # Constraint: v_above == v_below. This implies `unite(v_above, v_below)`.
                # Note: unite(a, b) handles the logic for a = b constraint.
                if not unite(v_above, v_below):
                    print(0) # Contradiction detected
                    return
                
                # Constraint: u_left == u_right. This implies `unite(u_left, u_right)`.
                if not unite(u_left, u_right):
                    print(0) # Contradiction detected
                    return

                # Additional constraint derived for Type B: v_rc != u_rc (vertical state != horizontal state for the specific edges of this cell)
                # Translates to: v_below != u_right => unite(v_below, neg(u_right)).
                # The variables `v_below_var_idx` and `u_right_var_idx` correspond to `v_rc` and `u_rc`.
                if not unite(v_below, u_right ^ 1):
                     print(0) # Contradiction detected
                     return

    # If all constraints are processed without contradiction, count the number of independent components.
    # Each independent component pair {Comp(x), Comp(neg x)} represents a degree of freedom (a binary choice).
    
    # Collect all unique roots to count the number of distinct sets (components).
    # Use a set data structure for efficient check of uniqueness.
    roots = set()
    for i in range(num_literals):
       roots.add(find_iterative(i)) 
    
    num_component_roots = len(roots)

    # The total number of components must be even, because literals come in pairs (x, neg x)
    # and x and neg x must end up in different components if no contradiction exists.
    # If num_component_roots is odd, something went wrong (or maybe it is possible if a variable is fixed to True/False?)
    # However, the DSU structure ensures pairs of components, so it should be even.
    # The number of independent choices = number of component pairs = total components / 2.
    num_pairs = num_component_roots // 2
    
    # The total number of valid assignments (solutions) is 2 raised to the power of the number of pairs.
    # Calculate this value modulo MOD.
    result = power(2, num_pairs)
    print(result)

# Read the number of test cases from standard input.
T = int(sys.stdin.readline())
# Iterate through each test case and call the solve function.
for _ in range(T):
    solve()