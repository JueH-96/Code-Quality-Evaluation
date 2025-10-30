# YOUR CODE HERE
import sys

# Function definition
def solve():
    # Read inputs using fast I/O
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    Q = int(sys.stdin.readline())

    # Convert string to list of characters for mutable operations
    # This allows O(1) time complexity for character modifications at specific indices.
    s_list = list(S)
    
    # Keep track of the operation index (0 to Q-1) of the last Type 1 modification for each character position.
    # Initialize with -1 to indicate no Type 1 modification has occurred yet for any position.
    last_modified_op_idx = [-1] * N 
    
    # Keep track of the last global case change operation (Type 2 or 3).
    # last_global_op_idx stores the index of the operation (0 to Q-1).
    # last_global_op_type stores the type (2 for lowercase, 3 for uppercase).
    # Initialize with -1 and 0 respectively, indicating no global operation has occurred yet.
    last_global_op_idx = -1 
    last_global_op_type = 0 # 0: none, 2: lowercase, 3: uppercase

    # Process each query/operation one by one.
    # Use a 0-based loop counter k for the operation index (from 0 to Q-1).
    for k in range(Q):
        # Read the current operation details from standard input.
        line = sys.stdin.readline().split()
        t = int(line[0]) # Operation type (1, 2, or 3)
        x = int(line[1]) # 1-based index for Type 1, unused and set to 0 otherwise
        c = line[2]      # Character for Type 1, unused and set to 'a' otherwise
        
        # The index k serves as the timestamp or identifier for the operation.
        op_idx = k 
        
        if t == 1:
            # Type 1: Change a specific character
            # Convert 1-based input index x to 0-based list index idx
            idx = x - 1 
            # Update the character at the specified index in the list
            s_list[idx] = c
            # Record the index k of this operation as the last Type 1 modification for this character position.
            last_modified_op_idx[idx] = op_idx 
            
        elif t == 2:
            # Type 2: Convert all uppercase letters in S to lowercase
            # Record the index k of this global lowercase operation. This overrides any previous global operation record.
            last_global_op_idx = op_idx
            last_global_op_type = 2
        
        elif t == 3:
            # Type 3: Convert all lowercase letters in S to uppercase
            # Record the index k of this global uppercase operation. This overrides any previous global operation record.
            last_global_op_idx = op_idx
            last_global_op_type = 3

    # After processing all Q operations, perform a final pass over the string.
    # This pass ensures that the effect of the *last* global operation (Type 2 or 3) is correctly applied
    # to characters that were not modified by a Type 1 operation *after* this last global operation.
    if last_global_op_type != 0: # Check if any global operation (Type 2 or 3) actually occurred during the Q operations.
        for j in range(N):
            # Check if the character at index j was last modified by a Type 1 operation *before* the last global operation occurred.
            # This includes characters that were never modified by any Type 1 operation (where last_modified_op_idx[j] remains -1).
            # The condition `last_modified_op_idx[j] < last_global_op_idx` correctly identifies these characters.
            if last_modified_op_idx[j] < last_global_op_idx:
                # If the character's last explicit modification (if any) predates the last global operation,
                # its case needs to be set according to that last global operation.
                if last_global_op_type == 2: # If the last global operation was Type 2 (lowercase)
                    s_list[j] = s_list[j].lower()
                elif last_global_op_type == 3: # If the last global operation was Type 3 (uppercase)
                    s_list[j] = s_list[j].upper()
            # If `last_modified_op_idx[j] >= last_global_op_idx`, it means the character at index j
            # was set by a Type 1 operation that occurred at the same time as or after the last global operation.
            # In this scenario, the character and its case as set by that Type 1 operation are final.
            # They should not be altered by the (potentially earlier or simultaneous) global operation.
            # Hence, we do nothing for these characters in this final pass.
    
    # Finally, join the list of characters back into a single string.
    final_string = "".join(s_list)
    # Print the resulting string to standard output.
    print(final_string)

# Call the solve function to execute the main logic of the program.
solve()

# END OF YOUR CODE HERE