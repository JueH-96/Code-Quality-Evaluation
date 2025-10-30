def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read inputs
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3].strip()
    
    # We interpret the wind directions by their vector shifts.
    # 'N': up (row -1), 'S': down (row +1), 'E': right (col +1), 'W': left (col -1)
    dirmap = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, 1),
        'W': (0, -1)
    }
    
    # Our simulation: At time 0, smoke is only at cell (0,0) and we say that
    # a "smoke particle" is originated at the campfire when the cell (0,0) is empty
    # after the wind move. In fact, if a smoke is generated then it can be tracked by
    # its "origin" time index j and its current position at time t+0.5 is just:
    #    T[t] - T[j]
    # where T[k] is the cumulative displacement after k wind moves.
    #
    # By simulation we deduce that a new smoke is generated at time t (after wind blow)
    # if and only if T[t] is a value that has not been seen before among T[0], T[1], …, T[t-1].
    # Let X be the set of indices of generation events. Initially X contains 0 (with T[0]=(0,0)).
    #
    # At time t+0.5 (after the wind blow at time t, but before any campfire addition),
    # the smoke positions come from every origin j in X with j < t, each located at:
    #       position = T[t] - T[j]
    # So cell (R, C) is occupied at time t+0.5 if and only if there is some j in X (with j < t)
    # such that:
    #       T[t] - T[j] = (R, C)
    # which is equivalent to
    #       T[j] = T[t] - (R, C)
    #
    # Notice that every time a coordinate appears for the first time among the cumulative
    # positions T[0], T[1], …, T[t] the campfire generates smoke. So for the purpose
    # of answering the query (whether smoke is present at (R,C) at time t+0.5),
    # it is enough to check if T[t] - (R,C) has occurred as one of T[0], …, T[t-1].
    
    # We'll maintain a set "seen" that holds all distinct T[j] (for generation events)
    # for j=0 (initially) and each time a new coordinate is encountered.
    seen = set()
    seen.add((0, 0))
    
    # cr, cc will store the cumulative displacement T[t]
    cr, cc = 0, 0
    
    # We accumulate our answers for each time step t (each corresponding to time t+0.5)
    # as explained above.
    ans = []
    for ch in S:
        dr, dc = dirmap[ch]
        cr += dr
        cc += dc
        # For the current t, we want to know: does there exist a j in seen (i.e. among T[0..t-1])
        # such that T[j] == (cr - R, cc - C)? If yes, then a smoke particle that was generated
        # in an earlier step has moved to (R,C) at time t+0.5.
        if (cr - R, cc - C) in seen:
            ans.append("1")
        else:
            ans.append("0")
        # If T[t] hasn't been seen before, then a new smoke particle is generated at the campfire,
        # so we add (cr, cc) to the set.
        if (cr, cc) not in seen:
            seen.add((cr, cc))
    
    sys.stdout.write("".join(ans))

if __name__ == '__main__':
    main()