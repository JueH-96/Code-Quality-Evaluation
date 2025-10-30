def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast parsing
    N = int(input_data[0])
    D = list(map(int, input_data[1:N+1]))
    idx = N+1
    L1, C1, K1 = map(int, input_data[idx:idx+3]); idx += 3
    L2, C2, K2 = map(int, input_data[idx:idx+3]); idx += 3

    # -------------------------------------------------------------------------
    # 1) Quick feasibility check: if any required D_i > maximum total coverage 
    #    (L1*K1 + L2*K2), it's outright impossible.
    # -------------------------------------------------------------------------
    max_cover = L1*K1 + L2*K2
    for length in D:
        if length > max_cover:
            print(-1)
            return

    # -------------------------------------------------------------------------
    # 2) For each section i, build a list of all (x1, x2) pairs (with 0 <= x1 <= K1, 
    #    0 <= x2 <= K2) such that x1*L1 + x2*L2 >= D[i].  
    #
    #    We only need to consider x1 up to either K1 or enough to cover 
    #    the whole D[i] alone.  That is x1 in [0.. min(K1, ceil(D[i]/L1))].
    #
    #    Then x2 = ceil( (D[i] - x1*L1)/L2 ) if that is > 0, else 0.
    #    If x2 <= K2, we record (x1,x2).
    #
    #    If for some i the coverage list is empty, answer is -1.
    # -------------------------------------------------------------------------
    coverage_sets = []
    for length in D:
        cover_list = []
        # The max x1 we need to consider is the smaller of K1 or just enough
        # to cover 'length' alone with type-1 sensors:
        # i.e. x1_max = min(K1, ceil(length/L1)) 
        # but we will just do integer loop carefully.
        if L1 != 0:
            x1_max = min(K1, (length + L1 - 1)//L1)
        else:
            # L1=0 would be nonsensical given constraints, but just be safe
            x1_max = 0

        found_any = False
        for x1 in range(x1_max+1):
            remain = length - x1*L1
            if remain <= 0:
                # This means x1*L1 >= length, so x2=0 is enough
                cover_list.append((x1, 0))
                found_any = True
                # Once we've found x1 s.t. x2=0, no need to try bigger x1
                # because that just uses more type-1 sensors (but still valid).
                # We do include them as well, because they are different usage.
                # However, we might as well continue to gather them, 
                # because in some global distribution we might prefer using 
                # more type-1 if type-2 is precious or vice versa.
                # So let's NOT break; we keep collecting all.
            else:
                # compute x2 needed
                x2 = (remain + L2 - 1)//L2
                if x2 <= K2:
                    cover_list.append((x1, x2))
                    found_any = True

        if not found_any:
            print(-1)
            return

        coverage_sets.append(cover_list)

    # -------------------------------------------------------------------------
    # 3) We will do a 2D "subset-sum"-style DP in (used1, used2) space, 
    #    but as a boolean.  The array feasible[a] will be a bitmask 
    #    for b=0..K2 (the b-th bit set means (a,b) is feasible).
    #
    #    Start with feasible[0] = 1 << 0 (meaning (0,0) is feasible).
    #    Then for each section i, we build nextFeasible via:
    #        nextFeasible[a + x1] |= (feasible[a] << x2),
    #    clipped by the limit b <= K2.
    #
    #    After processing all N sections, any (a,b) that is feasible 
    #    yields cost = a*C1 + b*C2; we pick the minimum.
    #
    #    If none feasible, answer = -1.
    # -------------------------------------------------------------------------

    # Python int can hold up to K2+1 bits easily, but we'll have to mask off.
    mask_limit = (1 << (K2+1)) - 1

    # feasible[a] = bitmask for b in 0..K2
    feasible = [0]* (K1+1)
    feasible[0] = 1  # (0,0) is feasible => bit 0 is set

    for i in range(N):
        cover_list = coverage_sets[i]
        next_feasible = [0]* (K1+1)
        for a in range(K1+1):
            old_mask = feasible[a]
            if old_mask == 0:
                continue
            for (x1, x2) in cover_list:
                new_a = a + x1
                if new_a <= K1:
                    shifted = (old_mask << x2) & mask_limit
                    next_feasible[new_a] |= shifted
        feasible = next_feasible

    # -------------------------------------------------------------------------
    # 4) Find the minimum cost among all (a,b) with feasible[a][b] == True
    # -------------------------------------------------------------------------
    ans = None
    for a in range(K1+1):
        mask = feasible[a]
        if mask == 0:
            continue
        # check which bits are set
        # while mask != 0, we can find the set bits
        # an easy way is to do a built-in method or popcount loop
        # but we must find each set bit's position.
        bpos = 0
        mm = mask
        while mm:
            # isolate lowest set bit
            lb = mm & (-mm)
            # index of that bit
            bpos = (lb).bit_length()-1  # if lb=1<<k, then bit_length()-1 = k
            cost = a*C1 + bpos*C2
            if ans is None or cost < ans:
                ans = cost
            mm ^= lb  # remove that bit from mm

    if ans is None:
        print(-1)
    else:
        print(ans)

# Don't forget to call main!
if __name__ == "__main__":
    main()