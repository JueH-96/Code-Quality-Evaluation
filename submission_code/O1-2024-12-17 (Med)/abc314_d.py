def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    # Parse inputs
    N = int(input_data[0])
    S = input_data[1]
    Q = int(input_data[2])

    # We'll keep track of:
    # 1) For each position, the time of the last single-character update (0 if none)
    # 2) The character used in that last update
    # 3) The time and type of the last global transformation (2 or 3), if any

    last_update_time = [0]*N
    last_update_char = ['']*N

    last_global_time = 0
    last_global_type = 0  # 2 => force-lower, 3 => force-upper, 0 => none

    idx = 3
    # Read the Q operations in chronological order i = 1..Q
    for i in range(1, Q+1):
        t = int(input_data[idx]); idx += 1
        x = int(input_data[idx]); idx += 1
        c = input_data[idx];     idx += 1

        if t == 1:
            # Single-character update
            # x is 1-based index of the character; store time i and character c
            pos = x - 1
            last_update_time[pos] = i
            last_update_char[pos] = c
        elif t == 2 or t == 3:
            # Global transformation (uppercase->lowercase or lowercase->uppercase)
            last_global_time = i
            last_global_type = t
        else:
            # Should not happen per problem statement
            pass

    # Function to apply the global transformation to a single character
    def transform_case(ch, op_type):
        if op_type == 2:
            # Convert to lowercase
            return ch.lower()
        else:
            # Convert to uppercase
            return ch.upper()

    # Build the final string
    result = [''] * N
    for i in range(N):
        sc_time = last_update_time[i]
        if sc_time == 0:
            # No single-character update
            base_char = S[i]
            if last_global_time > 0:
                # Apply the last global transform
                result[i] = transform_case(base_char, last_global_type)
            else:
                # No global transform either
                result[i] = base_char
        else:
            # There is a single-character update
            c = last_update_char[i]
            if sc_time > last_global_time:
                # This single update happened after the last global transform
                # So it takes precedence as-is
                result[i] = c
            else:
                # The single update is overshadowed by the last global transform
                result[i] = transform_case(c, last_global_type)

    print("".join(result))

# Do not forget to call main()!
if __name__ == "__main__":
    main()