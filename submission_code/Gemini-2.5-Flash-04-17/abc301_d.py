import sys

def solve():
    # Read input from stdin
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())

    n = len(S)
    # Store original S as a list to easily check which positions were originally '?'
    original_s_list = list(S)

    # Initialize final_config_list. This list will be modified to build the answer.
    # Start with replacing all '?' by '0' to get the minimum possible value.
    final_config_list = list(S)
    for i in range(n):
        if final_config_list[i] == '?':
            final_config_list[i] = '0'

    # Check if the minimum possible value (all '?' as '0') is greater than N.
    # If so, no value in T is <= N.
    # Convert the list back to string for int()
    min_val_str = "".join(final_config_list)
    min_val = int(min_val_str, 2)

    if min_val > N:
        print(-1)
        return

    # The minimum value (all '?' as '0') is a valid candidate <= N.
    # final_config_list currently represents this minimum value.
    # We now try to increase this value by changing some '?' from '0' to '1'
    # from most significant bit (left, index 0) to least significant bit (right, index n-1),
    # while ensuring the resulting number is <= N.

    for i in range(n):
        # If the original character at this position was '?', we have a choice ('0' or '1').
        # We try setting it to '1' to maximize the number.
        if original_s_list[i] == '?':
            
            # Construct a check string to see if setting bit i to '1' is feasible.
            # This string represents the smallest possible number we could get
            # if we set the current bit (at index i) to '1', given the decisions
            # already made for bits j < i.
            # It uses:
            # - The bits already fixed in final_config_list for positions k < i.
            # - '1' for the current position k = i.
            # - '0' for all subsequent positions k > i that were originally '?'.
            # - The original characters for subsequent positions k > i that were '0' or '1'.
            
            check_str_list = []
            for k in range(n):
                if k < i:
                    # Use the bit already decided and committed in final_config_list
                    check_str_list.append(final_config_list[k])
                elif k == i:
                    # Use '1' for the current position we are testing
                    check_str_list.append('1')
                else: # k > i
                    # If the original character was '?', assume '0' for the check (pessimistic after setting current bit to 1)
                    if original_s_list[k] == '?':
                        check_str_list.append('0')
                    else:
                        # Use the original fixed bit
                        check_str_list.append(original_s_list[k])

            check_str = "".join(check_str_list)
            potential_val_if_current_is_one = int(check_str, 2)

            # If this potential value (current bit = '1', future '?' = '0')
            # is less than or equal to N, it means setting the current bit to '1'
            # is feasible while staying <= N. To maximize the number, we commit
            # to setting this bit to '1' in our final_config_list.
            if potential_val_if_current_is_one <= N:
                 final_config_list[i] = '1'
            # else: setting this bit to '1' makes it too large, so it must remain '0'.
            # final_config_list[i] is already '0' from the initial setup, so nothing to do.


    # After iterating through all positions, final_config_list represents the binary
    # string of the greatest value in T less than or equal to N.
    final_val_str = "".join(final_config_list)
    final_val = int(final_val_str, 2)

    # Print the answer to stdout
    print(final_val)

# Execute the solve function
solve()