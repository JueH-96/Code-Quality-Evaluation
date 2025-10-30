def shiftDistance(s: str, t: str, nextCost: List[int], previousCost: List[int]) -> int:
    s = list(s)
    t = list(t)
    n = len(s)
    m = len(t)
    if n != m:
        return 0  # According to problem statement, s and t are same length
    
    # Precompute prefix sums for nextCost
    next_prefix = [0] * (26 + 1)
    for i in range(1, 27):
        next_prefix[i] = next_prefix[i-1] + nextCost[i-1]
    
    # Precompute prefix sums for previousCost
    previous_prefix = [0] * (26 + 1)
    for i in range(1, 27):
        previous_prefix[i] = previous_prefix[i-1] + previousCost[i-1]
    
    total = 0
    for i in range(n):
        sc = ord(s[i]) - ord('a')
        tc = ord(t[i]) - ord('a')
        
        if sc == tc:
            total += 0
            continue
        
        if tc > sc:
            # Compute cost via next shifts
            next_shift = next_prefix[sc + 1] - next_prefix[sc]
            total += next_shift
        else:
            # Compute cost via previous shifts
            prev_shift = previous_prefix[sc] - previous_prefix[tc + 1]
            total += prev_shift
    
    return total