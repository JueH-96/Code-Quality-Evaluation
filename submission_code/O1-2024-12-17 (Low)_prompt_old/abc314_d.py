def solve():
    import sys

    input_data = sys.stdin.read().strip().split()
    # index pointer for reading input_data
    ptr = 0

    # Read N and S
    N = int(input_data[ptr]); ptr += 1
    S = input_data[ptr]; ptr += 1

    # Read Q
    Q = int(input_data[ptr]); ptr += 1

    # Prepare arrays to track each character's last update
    last_update_time = [0]*N
    last_char = list(S)

    # Track the last global operation
    last_global_op_time = 0
    # 0 means none, 2 means "convert all to lowercase", 3 means "convert all to uppercase"
    last_global_op_type = 0

    # Process Q operations
    for i in range(1, Q+1):
        t_i = int(input_data[ptr]); ptr += 1
        x_i = int(input_data[ptr]); ptr += 1
        c_i = input_data[ptr]; ptr += 1

        if t_i == 1:
            # Update a single character
            idx = x_i - 1
            last_update_time[idx] = i
            last_char[idx] = c_i
        else:
            # Global operation
            # Update the last global operation time and type
            last_global_op_time = i
            last_global_op_type = t_i

    # Build the final string
    result = []
    for j in range(N):
        if last_update_time[j] > last_global_op_time:
            # The local update is newer than the global operation
            result.append(last_char[j])
        else:
            # Apply the latest global operation
            if last_global_op_type == 2:
                # to lowercase
                result.append(last_char[j].lower())
            elif last_global_op_type == 3:
                # to uppercase
                result.append(last_char[j].upper())
            else:
                # No global operation
                result.append(last_char[j])

    print("".join(result))

# Call solve()
solve()