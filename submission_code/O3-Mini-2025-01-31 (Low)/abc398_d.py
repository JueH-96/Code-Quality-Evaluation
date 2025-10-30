def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    R = int(data[1])
    C = int(data[2])
    S = data[3].strip()
    
    # Mapping each direction to its corresponding displacement.
    dmap = {'N': (-1, 0), 'S': (1, 0), 'E': (0, 1), 'W': (0, -1)}
    
    # Let P[k] be the position after k wind moves, starting with P[0]=(0,0).
    # Smoke generations: A generation is produced at time t (1-indexed) if after the wind move
    # (i.e. at time t + 0.5) cell (0,0) was empty, which exactly means that the new prefix 
    # P[t] has not been seen in any earlier generation.
    #
    # Therefore, if a generation was produced at time i (or initially at time 0)
    # then the smoke from that generation, at any later time t, will be located at:
    #   P[t] - P[i]  (where P[i] is the prefix sum when generation was created).
    # 
    # We want to determine for each t=1...N (i.e., at time t+0.5),
    # whether there exists a generation i (with 0 <= i <= t)
    # such that P[t] - P[i] == (R, C), i.e. P[i] == P[t] - (R, C).
    
    # We precompute the prefix positions on the fly while tracking the positions at which
    # smoke generations occurred.
    
    # Initialize:
    P_r, P_c = 0, 0   # P[0] = (0,0)
    gen_set = {(0, 0)}  # generation exists at time 0 by default.
    
    # Build result as a list of characters.
    result_chars = []
    
    # Process each wind move t=1..N
    for i in range(1, N + 1):
        # Update prefix position using current wind.
        dr, dc = dmap[S[i - 1]]
        P_r += dr
        P_c += dc
        current_pos = (P_r, P_c)
        
        # According to the rules, if cell (0,0) did not have smoke just after the wind
        # move in this time step, then new smoke is generated at (0,0).
        # Notice that cell (0,0) ends up having smoke if there is some earlier generation j
        # such that P[i]-P[j]==(0,0) i.e. P[i]==P[j]. Thus, if current_pos is already one of
        # the generation positions, no new generation occurs.
        if current_pos not in gen_set:
            gen_set.add(current_pos)
        
        # Now, at time t+0.5, consider all smoke particles:
        # Each smoke that started at a generation position P[j] (with j <= i) is at:
        #   P[i] - P[j]
        # To have smoke at (R, C), we need some j with:
        #   P[i] - P[j] = (R, C)
        # i.e. P[j] = P[i] - (R, C).
        needed = (P_r - R, P_c - C)
        result_chars.append('1' if needed in gen_set else '0')
    
    sys.stdout.write("".join(result_chars))

if __name__ == '__main__':
    main()