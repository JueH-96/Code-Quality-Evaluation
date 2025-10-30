def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    R = int(input_data[1])
    C = int(input_data[2])
    S = input_data[3].strip()
    
    # Mapping of wind directions to (row, col) displacement.
    delta = {
        'N': (-1, 0),
        'S': (1, 0),
        'E': (0, 1),
        'W': (0, -1)
    }
    
    # At time 0, smoke is at (0,0) due to the campfire.
    # We will store the cumulative wind sum from time 0 up to time t
    # for each smoke that was generated. The smoke at generation time k (with
    # cumulative sum S_k) will have its position computed at a later time t as:
    #   position = S_t - S_k.
    # Note: S_0 = (0,0) is always present because of the initial campfire.
    
    # When wind blows at time t, we update the cumulative sum S_t.
    # After moving all smoke, if (0,0) is unoccupied then new smoke is added
    # i.e. we add S_t to our set of "generated" sums.
    # A smoke particle exists at (R, C) at time t + 0.5 if for some previously
    # generated cumulative sum g in our generation set, we have:
    #   S_t - g = (R, C)
    # which is equivalent to: g = (S_t - (R, C)).
    
    # Sgen keeps track of cumulative sums corresponding to the times when new smoke was generated.
    Sgen = set()
    Sgen.add((0, 0))  # time 0: campfire
    cur_r, cur_c = 0, 0  # cumulative sum S_t
    
    result = []
    for ch in S:
        dr, dc = delta[ch]
        cur_r += dr
        cur_c += dc
        
        # Check if (0,0) is occupied (i.e. exists some generation with g such that S_t - g = (0,0) <=> g = S_t).
        # If not, then add a new smoke particle, which is equivalent to adding S_t.
        if (cur_r, cur_c) not in Sgen:
            Sgen.add((cur_r, cur_c))
        
        # Smoke exists at (R, C) if there exists a previously generated smoke particle at position g
        # such that S_t - g = (R, C). Rearranging gives g = (cur_r - R, cur_c - C).
        if (cur_r - R, cur_c - C) in Sgen:
            result.append('1')
        else:
            result.append('0')
    
    sys.stdout.write("".join(result))
    
if __name__ == "__main__":
    main()