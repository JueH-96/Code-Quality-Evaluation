import sys

def solve():
    # Read N
    N = int(sys.stdin.readline())
    
    # Read the initial string S and convert it to a list of characters for mutability.
    S = sys.stdin.readline().strip()
    s_chars = list(S)
    
    # Read Q
    Q = int(sys.stdin.readline())

    # last_modified_op_idx[i] will store the 0-indexed operation number
    # of the *last* type 1 operation that modified the character at s_chars[i].
    # Initialize with -1 to indicate not modified by a type 1 op yet, or modified before any global case op.
    last_modified_op_idx = [-1] * N

    # last_case_op_type stores the type of the last global case conversion operation (2 for lower, 3 for upper).
    # Initialize with 0 (or any other value) to indicate no global conversion operations yet.
    last_case_op_type = 0 
    
    # last_case_op_idx stores the 0-indexed operation number of the *last* global case conversion operation.
    # Initialize with -1 to indicate no global conversion operations yet.
    last_case_op_idx = -1 

    # Process Q operations
    for i in range(Q):
        line = sys.stdin.readline().split()
        t = int(line[0])
        
        if t == 1:
            # Type 1: Change x_i-th character to c_i.
            # x_i is 1-based, convert to 0-based index.
            x = int(line[1])
            c = line[2]
            s_chars[x-1] = c
            # Record that this character was last modified by this operation (i).
            last_modified_op_idx[x-1] = i
        else:
            # Type 2 (convert to lowercase) or Type 3 (convert to uppercase).
            # We only care about the *last* such operation's type and its index.
            last_case_op_type = t
            last_case_op_idx = i

    # After processing all operations, perform the final case conversion.
    # This loop only runs if there was at least one type 2 or type 3 operation.
    if last_case_op_type == 2: # The last global conversion was to lowercase
        for i in range(N):
            # If the character at s_chars[i] was last modified by a type 1 operation
            # *before* the last global case conversion (last_modified_op_idx[i] < last_case_op_idx),
            # or if it was never modified by a type 1 operation (last_modified_op_idx[i] is -1),
            # then apply the global lowercase conversion.
            # Otherwise, it was modified *after* or *at the same time* as the last global
            # conversion, so its case is already determined by the character assigned by type 1.
            if last_modified_op_idx[i] < last_case_op_idx:
                s_chars[i] = s_chars[i].lower()
    elif last_case_op_type == 3: # The last global conversion was to uppercase
        for i in range(N):
            # Similar logic for uppercase.
            if last_modified_op_idx[i] < last_case_op_idx:
                s_chars[i] = s_chars[i].upper()

    # Join the list of characters back into a string and print it.
    sys.stdout.write("".join(s_chars) + "
")

# Call the solve function to run the program.
solve()