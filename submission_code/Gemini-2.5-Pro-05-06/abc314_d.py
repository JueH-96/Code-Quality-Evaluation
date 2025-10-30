import sys

def main():
    N = int(sys.stdin.readline())
    S_str = sys.stdin.readline().strip()
    # S_chars will store the characters of the string.
    # It's initialized with the input string S.
    # For type 1 operations, characters in S_chars are updated directly.
    # The case of these characters might be changed later during finalization.
    S_chars = list(S_str)
    
    Q = int(sys.stdin.readline())

    # last_type1_op_time[i] stores the operation number (1-indexed, from 1 to Q) 
    # of the last type 1 operation that modified the character at index i.
    # Initialized to 0, indicating that no type 1 operation has affected this character yet,
    # so it still holds its initial value from S_str.
    last_type1_op_time = [0] * N
    
    # Store the operation number and type of the chronologically last global case-change operation.
    last_global_op_num = 0 # Operation number (1 to Q)
    # last_global_op_type:
    #   0: No global case-change operation has occurred yet / is relevant.
    #   2: The last relevant global operation was type 2 (convert all to lowercase).
    #   3: The last relevant global operation was type 3 (convert all to uppercase).
    last_global_op_type = 0 

    # Process queries one by one
    for op_num in range(1, Q + 1):
        line = sys.stdin.readline().split()
        op_type = int(line[0])
        
        # x_param and c_param are read based on input format.
        # They are only used if op_type == 1.
        # For op_type == 1, x_param is 1-indexed character position.
        # For op_type != 1, problem states x_param = 0, c_param = 'a' (dummy values).
        x_param = int(line[1]) 
        c_param = line[2]

        if op_type == 1:
            # Type 1: Change a specific character
            char_idx = x_param - 1 # Convert to 0-indexed for list access
            S_chars[char_idx] = c_param
            # Record the time (operation number) of this modification
            last_type1_op_time[char_idx] = op_num
        elif op_type == 2:
            # Type 2: Convert all to lowercase
            # This becomes the new "last global operation"
            last_global_op_type = 2
            last_global_op_num = op_num
        else: # op_type == 3
            # Type 3: Convert all to uppercase
            # This becomes the new "last global operation"
            last_global_op_type = 3
            last_global_op_num = op_num

    # After all queries are processed, finalize the string S_chars.
    # This step applies the effect of the last global case-change operation,
    # but only to characters that were not modified by a type 1 operation *after* it.
    
    if last_global_op_type != 0: # Check if any global operation actually happened
                                 # and is therefore relevant for finalization.
        for i in range(N):
            # A character S_chars[i] needs its case adjusted by the last global operation if:
            # 1. It's an initial character never touched by a type 1 op (last_type1_op_time[i] == 0).
            #    In this case, 0 < last_global_op_num is true.
            # 2. It was last touched by a type 1 op *before* the last global op
            #    (last_type1_op_time[i] < last_global_op_num).
            # In summary, if last_type1_op_time[i] < last_global_op_num.
            if last_type1_op_time[i] < last_global_op_num:
                if last_global_op_type == 2: # Convert to lowercase
                    S_chars[i] = S_chars[i].lower()
                else: # last_global_op_type == 3, convert to uppercase
                    S_chars[i] = S_chars[i].upper()
            # If last_type1_op_time[i] >= last_global_op_num, this means the character
            # S_chars[i] was set by a type 1 operation at or after the last global operation.
            # Its case is determined by that type 1 operation, so S_chars[i] is already correct.
            # No change needed for S_chars[i] in this case.
    
    # Print the final string
    sys.stdout.write("".join(S_chars) + "
")

if __name__ == '__main__':
    main()