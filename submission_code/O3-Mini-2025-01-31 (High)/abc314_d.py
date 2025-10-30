def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    pos = 0
    N = int(data[pos])
    pos += 1
    S = list(data[pos])
    pos += 1
    Q = int(data[pos])
    pos += 1

    # For each index, we'll record the op-index of the last type1 update.
    # Initial characters get a timestamp of -1.
    last_update = [-1] * N

    # global_timestamp and global_mode track the most recent global conversion.
    # global_mode: 0 = none, 1 = convert to lowercase, 2 = convert to uppercase.
    global_timestamp = -1
    global_mode = 0

    op_index = 0  # This will be our counter for operations.
    for _ in range(Q):
        t = int(data[pos])
        pos += 1
        if t == 1:
            # Operation type 1: update S at a given position.
            x = int(data[pos])
            pos += 1
            c = data[pos]
            pos += 1
            S[x - 1] = c
            last_update[x - 1] = op_index
        elif t == 2:
            # Operation type 2: convert all uppercase letters in S to lowercase.
            pos += 2  # Skip dummy arguments.
            global_timestamp = op_index
            global_mode = 1
        else:
            # Operation type 3: convert all lowercase letters in S to uppercase.
            pos += 2  # Skip dummy arguments.
            global_timestamp = op_index
            global_mode = 2
        op_index += 1

    # When printing the final string, for each index:
    # If the last individual update happened before the last global op (i.e. last_update[i] < global_timestamp),
    # then that index is still affected by the global conversion.
    res_chars = []
    if global_mode == 0:
        # No global conversion ever happened.
        res_chars = S
    else:
        if global_mode == 1:
            # Global conversion type 2: uppercase letters to lowercase.
            for i, ch in enumerate(S):
                if last_update[i] < global_timestamp:
                    res_chars.append(ch.lower())
                else:
                    res_chars.append(ch)
        else:  # global_mode == 2, convert to uppercase.
            for i, ch in enumerate(S):
                if last_update[i] < global_timestamp:
                    res_chars.append(ch.upper())
                else:
                    res_chars.append(ch)
    sys.stdout.write("".join(res_chars))

if __name__ == '__main__':
    main()