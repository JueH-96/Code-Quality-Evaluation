def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    
    N = int(next(it))
    S = list(next(it).strip())
    Q = int(next(it))
    
    # For each position, record the time stamp (starting from 0 for "initial")
    # of the last op1 (character update) affecting that position.
    last_time = [0] * N
    
    # These track the last "global" operation (op2 or op3) that affected everyone.
    # global_last_time is the time stamp of that op.
    # global_last_type is 2 (convert to lowercase) or 3 (convert to uppercase).
    global_last_time = -1
    global_last_type = None

    current_time = 0
    for _ in range(Q):
        current_time += 1
        t = int(next(it))
        x = int(next(it))
        c = next(it)
        if t == 1:
            # Operation type1: update the x-th character (x is 1-indexed).
            idx = x - 1
            S[idx] = c
            last_time[idx] = current_time
        elif t == 2 or t == 3:
            # Global operations (op2: to lower, op3: to upper).
            # We simply record the time and type; later, we will apply this global conversion
            # to every index that did not have a later op1 update.
            global_last_time = current_time
            global_last_type = t

    # At the end, for each index we must decide: does the final character come
    # from a local update that occurred after the most recent global op or not?
    # If not, then the effect of the (last) global op should be applied.
    if global_last_time != -1:
        # Choose a transformation function according to the global op type.
        # op2 means "convert all uppercase letters to lowercase" (i.e. use str.lower)
        # op3 means "convert all lowercase letters to uppercase" (i.e. use str.upper)
        if global_last_type == 2:
            transform = str.lower
        else:  # global_last_type == 3
            transform = str.upper

        # For each index, if its last op1 time was strictly before the last global op,
        # then the final character is the one stored but converted by the global op.
        for i in range(N):
            if last_time[i] < global_last_time:
                S[i] = transform(S[i])
                
    sys.stdout.write("".join(S))
    
if __name__ == '__main__':
    main()