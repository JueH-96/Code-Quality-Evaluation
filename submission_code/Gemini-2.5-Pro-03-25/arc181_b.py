import sys

# Set higher recursion depth for safety, although iterative find is used.
# sys.setrecursionlimit(2 * (10**5 + 10)) 

def solve():
    """Solves a single test case."""
    
    S = sys.stdin.readline().strip()
    X = sys.stdin.readline().strip()
    Y = sys.stdin.readline().strip()

    # Handle the special 'empty' string representation used in Sample 2
    if S == 'empty': 
        S = '' 
    
    # Count occurrences of '0' and '1' in X and Y
    N0X = X.count('0')
    N1X = len(X) - N0X # Number of '1's in X
    N0Y = Y.count('0')
    N1Y = len(Y) - N0Y # Number of '1's in Y

    LS = len(S) # Length of string S

    # Calculate differences in counts
    # DeltaN0 = N0(X) - N0(Y)
    # DeltaN1 = N1(Y) - N1(X)  -- Note the order for N1 difference
    DeltaN0 = N0X - N0Y
    DeltaN1 = N1Y - N1X 

    # Case 1: DeltaN1 is 0
    # This means N1(X) = N1(Y), the number of T blocks must be the same.
    if DeltaN1 == 0:
        # The length equality condition becomes: DeltaN0 * LS = 0 * k = 0.
        if S == '':
             # If S is empty, LS = 0. The condition 0 = 0 always holds.
             # f(eps, T, X) = T^N1X. f(eps, T, Y) = T^N1Y.
             # Since N1X = N1Y, they are equal for any T (including empty T).
             print("Yes")
             return

        # If S is not empty, LS > 0.
        # Need DeltaN0 * LS = 0, which implies DeltaN0 = 0.
        if DeltaN0 == 0:
             # If DeltaN0 = 0, then N0X = N0Y.
             # Both counts match: N0X = N0Y and N1X = N1Y. This implies |X| = |Y|.
             # A valid T exists. For example, T=S works:
             # f(S, S, X) results in |X| blocks, all S. String is S^|X|.
             # f(S, S, Y) results in |Y| blocks, all S. String is S^|Y|.
             # Since |X|=|Y|, the strings are equal.
             # T=empty also works if N0X = N0Y, as f(S, eps, X) = S^N0X and f(S, eps, Y) = S^N0Y.
             print("Yes")
        else:
             # DeltaN0 != 0. The length condition DeltaN0 * LS = 0 fails because LS > 0.
             # No T can satisfy length equality.
             print("No")
        return

    # Case 2: DeltaN1 is not 0
    
    # If S is empty (LS = 0):
    if LS == 0: 
        # The length condition becomes DeltaN0 * 0 = DeltaN1 * k => 0 = DeltaN1 * k.
        # Since DeltaN1 != 0, this requires k = 0.
        # So T must be the empty string.
        # Check if T=empty works:
        # f(eps, eps, X) = eps^(N1X) = eps
        # f(eps, eps, Y) = eps^(N1Y) = eps.
        # The strings are always equal (both empty).
        print("Yes")
        return

    # If S is not empty (LS > 0):
    # The length equality condition is DeltaN0 * LS = DeltaN1 * k.
    # We need to solve for k (length of T). k = (DeltaN0 * LS) / DeltaN1.
    
    numerator = DeltaN0 * LS
    denominator = DeltaN1
    
    # Check if k is an integer. The numerator must be divisible by the denominator.
    if numerator % denominator != 0:
        print("No") # k is not an integer, impossible length for T.
        return

    k = numerator // denominator # Required length of T

    # Check if k is non-negative.
    if k < 0:
        print("No") # Required length of T is negative, impossible.
        return
    
    # Check if k is 0.
    if k == 0:
        # If k=0 is required, T must be the empty string. Check if T=empty works.
        # f(S, eps, X) = S^N0X (S repeated N0X times)
        # f(S, eps, Y) = S^N0Y (S repeated N0Y times)
        # We need S^N0X == S^N0Y. Since S is not empty, this holds if and only if N0X == N0Y.
        # Let's verify if N0X == N0Y must hold if k=0 is required.
        # The requirement k=0 comes from the length equation: DeltaN0 * LS = DeltaN1 * k = DeltaN1 * 0 = 0.
        # Since LS > 0, this implies DeltaN0 = 0.
        # DeltaN0 = N0X - N0Y. So DeltaN0 = 0 means N0X = N0Y.
        # The condition N0X = N0Y is indeed satisfied. Thus T=empty works.
        print("Yes")
        return

    # Case 3: k > 0. The required length of T is positive.
    # We need to check if there exists a string T of length k satisfying the equality.
    # Use simulation combined with Disjoint Set Union (DSU) to track constraints on T.
    
    parent = list(range(k)) # DSU parent array
    char_val = [None] * k   # Stores determined character (ASCII value) for the root of each set, None if undetermined.
    
    # Iterative find operation with path compression for DSU
    def find(i):
        root = i
        stack = []
        while parent[root] != root:
            stack.append(root)
            root = parent[root]
        # Path compression loop
        while stack:
             curr = stack.pop()
             parent[curr] = root
        return root

    conflict_found = False # Flag to indicate if an inconsistency is found

    # Union operation for DSU: Merges the sets containing i and j.
    # Checks for character conflicts if both sets already have determined characters.
    def union(i, j):
        nonlocal conflict_found
        if conflict_found: return # Stop if conflict already found

        root_i = find(i)
        root_j = find(j)
        
        if root_i != root_j: # Only merge if they are in different sets
            char_i = char_val[root_i] # Character assigned to root of i's set
            char_j = char_val[root_j] # Character assigned to root of j's set

            # Check for conflict
            if char_i is not None and char_j is not None:
                if char_i != char_j:
                    conflict_found = True # Conflict detected!
                    return

            # Determine the character for the merged set
            new_char = char_i if char_i is not None else char_j
            
            # Perform the merge (simple merge: make root_j child of root_i)
            # Union by rank/size could be used for better theoretical complexity but usually not needed in practice
            parent[root_j] = root_i
            # Update the character value of the new root
            char_val[root_i] = new_char 
    
    # Set_char operation for DSU: Assigns a character value `char_code` to the set containing index `i`.
    # Checks for conflict if the set already has a different character assigned.
    def set_char(i, char_code):
      nonlocal conflict_found
      if conflict_found: return # Stop if conflict already found

      root_i = find(i)
      current_char = char_val[root_i] # Current character assigned to the root

      if current_char is not None:
          # If character already assigned, check for conflict
          if current_char != char_code:
              conflict_found = True # Conflict detected!
      else:
          # If no character assigned yet, assign the new character
          char_val[root_i] = char_code

    
    # Precompute ASCII values for S for efficiency
    S_codes = [ord(c) for c in S] 

    # Simulation state variables
    idx_X = 0 # Current block index in X
    idx_Y = 0 # Current block index in Y
    
    char_in_block_X = 0 # Current character index within the block X[idx_X]
    char_in_block_Y = 0 # Current character index within the block Y[idx_Y]

    # Simulate the string comparison character by character implicitly
    # The simulation proceeds by comparing blocks and skipping identical parts
    while idx_X < len(X) and idx_Y < len(Y):
        if conflict_found: break # Exit loop early if conflict found

        # Determine block types ('S' or 'T') and lengths
        is_S_X = (X[idx_X] == '0')
        is_S_Y = (Y[idx_Y] == '0')
        
        len_block_X = LS if is_S_X else k # Length of current block in X sequence
        len_block_Y = LS if is_S_Y else k # Length of current block in Y sequence

        # Calculate remaining lengths in current blocks
        rem_len_X = len_block_X - char_in_block_X
        rem_len_Y = len_block_Y - char_in_block_Y

        # Determine how many characters to process in this step
        step = min(rem_len_X, rem_len_Y) 

        # Compare the segments of length 'step' based on block types
        if is_S_X and is_S_Y:
             # S vs S block: Characters are identical (S[p] == S[p]). Always matches.
             # No constraints added. Just advance pointers.
             pass 
        elif not is_S_X and not is_S_Y:
             # T vs T block: Need T[p_X] == T[p_Y] for corresponding character indices p_X, p_Y.
             # Use DSU to enforce equality constraints T[char_idx_X] == T[char_idx_Y].
             for p_offset in range(step):
                 union(char_in_block_X + p_offset, char_in_block_Y + p_offset)
                 if conflict_found: break # Check conflict after each union
        elif is_S_X and not is_S_Y:
            # S vs T block: Need S[p_X] == T[p_Y].
            # Use DSU to assign character values to T based on S.
             for p_offset in range(step):
                 set_char(char_in_block_Y + p_offset, S_codes[char_in_block_X + p_offset])
                 if conflict_found: break # Check conflict after each set_char
        else: # not is_S_X and is_S_Y
             # T vs S block: Need T[p_X] == S[p_Y].
             # Use DSU to assign character values to T based on S.
             for p_offset in range(step):
                 set_char(char_in_block_X + p_offset, S_codes[char_in_block_Y + p_offset])
                 if conflict_found: break # Check conflict after each set_char
        
        if conflict_found: break # Exit outer loop if inner loop found conflict
            
        # Advance character pointers within blocks
        char_in_block_X += step
        char_in_block_Y += step

        # Move to the next block if the current one is finished
        if char_in_block_X == len_block_X:
            idx_X += 1
            char_in_block_X = 0 # Reset character index for the new block
        
        if char_in_block_Y == len_block_Y:
            idx_Y += 1
            char_in_block_Y = 0 # Reset character index for the new block
            
    # Final decision based on whether a conflict was found
    if conflict_found:
        print("No") # An inconsistency was detected, no such T exists.
    else:
        # If no conflict was found, it means all constraints are consistent.
        # Since the total lengths of f(S, T, X) and f(S, T, Y) must match (due to how k was derived),
        # reaching the end of simulation without conflict implies both idx_X and idx_Y reached ends.
        # A valid T (possibly partially determined) exists.
        # We don't need to construct T, just know it exists.
        print("Yes")

# Read the number of test cases
T_cases = int(sys.stdin.readline())
# Process each test case
for _ in range(T_cases):
    solve()