def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    # Parse the first three lines
    N = int(input_data[0])
    S = input_data[1]
    Q = int(input_data[2])
    
    # We will keep for each position the tuple (last_update_time, current_char).
    # Initially, last_update_time=0 (meaning "before any operation"), and current_char = S[i].
    updates = [(0, ch) for ch in S]
    
    # We track the index (1-based) of the *last* case-conversion operation
    last_transform_index = 0
    last_transform_type = 0  # 2 => to lower, 3 => to upper, 0 => none so far
    
    # The operations start from input_data[3..]
    # Each operation line has: t_i, x_i, c_i (but x_i,c_i only matter if t_i=1)
    # We'll parse them in order:
    idx = 3
    for i in range(1, Q+1):
        t = int(input_data[idx]); idx += 1
        x = int(input_data[idx]); idx += 1
        c = input_data[idx]; idx += 1
        
        if t == 1:
            # update character at position x-1 to c
            # record that the last update to this position happened at operation i
            updates[x-1] = (i, c)
        else:
            # t=2 => to lower, t=3 => to upper
            last_transform_index = i
            last_transform_type = t
    
    # After processing all Q operations,
    # for each position j, check if its last update time < last_transform_index.
    # If so, apply the last transformation to the character.
    
    result = []
    to_lower = (last_transform_type == 2)
    to_upper = (last_transform_type == 3)
    
    for (uptime, ch) in updates:
        if last_transform_index > uptime:
            if to_lower:
                ch = ch.lower()
            elif to_upper:
                ch = ch.upper()
        # else we keep ch as is
        result.append(ch)
    
    print("".join(result))

# Don't forget to call main().
main()