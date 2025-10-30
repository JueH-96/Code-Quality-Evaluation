def solve():
    import sys
    input_data=sys.stdin.read().strip().split()
    N=int(input_data[0])
    D=int(input_data[1])
    xs=[int(x) for x in input_data[2::2]]
    ys=[int(y) for y in input_data[3::2]]

    #----------------------------------------------------------------
    # We want all integer (x, y) such that sum_i(|x - x_i| + |y - y_i|) <= D.
    #
    # Let F(x) = sum_i |x - x_i|, and G(y) = sum_i |y - y_i|.
    # Then the condition is F(x) + G(y) <= D.
    #
    # The set of feasible (x,y) is all x with F(x) <= D, and for each such x,
    # all y with G(y) <= D - F(x).
    #
    # The number of solutions equals:
    #   sum_{x : F(x) <= D} [ number_of_integers y s.t. G(y) <= D - F(x) ].
    #
    # However, directly iterating over all x in a large range could be huge.
    # But we exploit the fact that F(x) is piecewise-linear in x, with a unique
    # minimum around the median of x_i, and it increases as we go away from that median.
    # Once F(x) exceeds D, it will not come back below D (F is convex in x).
    #
    # Algorithm sketch:
    #   1. Sort x_i in Xarr. We'll find a "center" x0 (take the median) and compute F(x0) in O(N).
    #   2. We'll then step left from x0 downwards and right from x0 upwards, updating F(x) incrementally:
    #        F(x+1) = F(x) + countLeft(x) - (N - countLeft(x)) = F(x) + 2*countLeft(x) - N,
    #      where countLeft(x) = number of i with x_i <= x.
    #      Because moving x -> x+1 increases the distance by 1 for each x_i <= x, and decreases it by 1 for each x_i > x.
    #      We stop in each direction as soon as F(x) > D because beyond that it will only grow further.
    #      Collect freqF[val] = (count of integer x that give F(x) = val).
    #
    #   3. Repeat the same procedure for G(y) with sorted y_i to get freqG[val].
    #
    #   4. Then we need to sum over all pairs (sx, sy) with sx + sy <= D of freqF[sx]*freqG[sy].
    #      We'll do this via a two-pointer approach on the sorted keys of freqF and freqG.
    #
    # Complexity:
    #   - Finding F(x0) takes O(N).
    #   - Sweeping left/right for x to build freqF can take up to ~2D steps in the worst case.
    #   - Same for freqG in y.
    #   - Then combining with two-pointer is O(|freqF| + |freqG|) which is at most ~4D.
    #   - In worst case D can be 1e6, so we can have about 2e6 steps to the left/right, summing to up to 4e6 + overhead.
    #     This is large but can sometimes be done in optimized Python if carefully implemented (especially in faster languages it is safer).
    #     We will implement as efficiently as possible in Python.
    #
    # Edge cases:
    #   - N=1: Then F(x) = |x - x1|. That can still produce up to 2D+1 feasible x. We'll handle it the same way.
    #   - D=0: Then we only count (x, y) if sum_i(|x - x_i| + |y - y_i|)=0 => x= x_i and y= y_i for all i, i.e. only possible if all x_i are the same and all y_i are the same. Otherwise 0.
    #
    # Let's implement.
    #----------------------------------------------------------------

    # Special quick check: if D=0, the only way sum of distances is 0 is if all x_i are the same and all y_i are the same.
    # Then the answer would be 1 if all x_i == x_1 and all y_i == y_1, else 0.
    if D == 0:
        # Check if all (x_i, y_i) are the same
        # Constraint says (x_i, y_i) are distinct for i != j, so the answer must be 0 unless N=1
        # Actually the problem states: (x_i, y_i) != (x_j, y_j) for i!=j, so for N>1 there is no single (x,y) that meets all exactly.
        # => answer=0 if N>1, else 1 if N=1.
        print(1 if N==1 else 0)
        return

    # Sort x and y
    Xarr = sorted(xs)
    Yarr = sorted(ys)

    # Prefix sums for x and y to quickly compute F(x0) or G(y0)
    import itertools
    prefixX = [0]*(N+1)  # prefixX[i] = sum of Xarr[:i]
    for i in range(N):
        prefixX[i+1] = prefixX[i] + Xarr[i]
    prefixY = [0]*(N+1)
    for i in range(N):
        prefixY[i+1] = prefixY[i] + Yarr[i]

    # Helper to compute sum_i |val - Xarr[i]| in O(N) (used once for initial)
    # We'll do it by splitting at a rank r, val in [Xarr[r], Xarr[r+1]] range or so,
    # but here we just do it for a chosen val. We'll find position with binary search.
    def sum_abs_differences_x(val):
        # pos is number of elements <= val
        # (We can do a standard library bisect in Xarr)
        import bisect
        pos = bisect.bisect_right(Xarr, val)
        # sum of all to left: sum_{i<pos} (val - Xarr[i])
        left_sum = val*pos - prefixX[pos]
        # sum of all to right: sum_{i>=pos} (Xarr[i] - val)
        right_count = N - pos
        right_sum = (prefixX[N] - prefixX[pos]) - val*right_count
        return left_sum + right_sum

    def sum_abs_differences_y(val):
        import bisect
        pos = bisect.bisect_right(Yarr, val)
        left_sum = val*pos - prefixY[pos]
        right_count = N - pos
        right_sum = (prefixY[N] - prefixY[pos]) - val*right_count
        return left_sum + right_sum

    # We'll pick x0 = median of Xarr (for odd N) or one median if even N
    # Actually we can pick Xarr[N//2], that is a valid "median" index (0-based).
    x0 = Xarr[N//2]
    F0 = sum_abs_differences_x(x0)
    # We also need countLeft(x0).
    import bisect
    cL0 = bisect.bisect_right(Xarr, x0)  # number of xi <= x0
    # We'll store (Fvalue -> count of x).
    from collections import defaultdict
    freqF = defaultdict(int)

    # We'll do two expansions: left side and right side (including x0).
    # 1) put x = x0 in freqF
    freqF[F0] += 1

    # We'll expand to the right:
    # We keep track of current x, F(x), countLeft(x).
    cur_x = x0
    cur_F = F0
    cur_cL = cL0
    while True:
        nxt_x = cur_x + 1
        # F(nxt_x) = F(cur_x) + (number of xi <= cur_x) - (number of xi > cur_x)
        #          = F(cur_x) + 2*countLeft(cur_x) - N
        nxt_F = cur_F + 2*cur_cL - N
        if nxt_F > D:
            break
        freqF[nxt_F] += 1
        # update countLeft for nxt_x
        # how many xi <= nxt_x?
        # we can move a pointer forward in Xarr if needed:
        # We'll do a quick while approach, but that could be O(N). Instead do a bisect for x+1:
        # But we only move one step at a time, so let's do a pointer-based approach.
        # However, we can do: new_cL = bisect_right(Xarr, nxt_x).
        new_cL = bisect.bisect_right(Xarr, nxt_x)
        cur_x, cur_F, cur_cL = nxt_x, nxt_F, new_cL

    # We'll expand to the left similarly:
    cur_x = x0
    cur_F = F0
    cur_cL = cL0
    while True:
        nxt_x = cur_x - 1
        # F(nxt_x) = F(cur_x) + (number of xi >= cur_x) - (number of xi < cur_x)
        # but let's re-derive carefully. If we move from x to x-1:
        # difference = sum_i |(x-1)-xi| - |x - xi|
        # for xi <= x-1 => distance changes by +1
        # for xi >= x   => distance changes by -1
        # So how many xi >= x? That is N - (countLeft(x)), but note countLeft(x) counts xi <= x
        # The difference = (countLeft(x-1)) - (N - countLeft(x-1)), but we have cL for x not x-1
        # It's easier to do symmetrical approach:
        # F(x-1) = F(x) + (#xi > x-1) - (#xi <= x-1)
        # We want countLeft(x-1) = # xi <= x-1
        nxt_cL = bisect.bisect_right(Xarr, nxt_x)  # # xi <= x-1
        # difference = 2*(N - nxt_cL) - N = 2N - 2nxt_cL - N = N - 2*nxt_cL
        # Wait carefully: When we go from x to x-1, for each i with xi <= x-1, distance increases by 1; for each i with xi > x, distance decreases by 1; for xi in (x-1, x], handle carefully. 
        # An easier consistent formula: F(x0+1) - F(x0) = 2*cL(x0) - N. 
        # So F(x0-1) - F(x0) = ?
        # By symmetry, F(x-1) = F(x) - [2*cL(x-1) - N].
        # Because cL(x-1) = # xi <= x-1. 
        # So let's define diff = 2*cL(x-1) - N. Then F(x-1) = F(x) + -diff = F(x) - diff.
        diff = 2*nxt_cL - N
        nxt_F = cur_F - diff
        if nxt_F > D:
            break
        freqF[nxt_F] += 1
        cur_x, cur_F, cur_cL = nxt_x, nxt_F, nxt_cL

    # Repeat the same for G(y).
    y0 = Yarr[N//2]
    G0 = sum_abs_differences_y(y0)
    cL0y = bisect.bisect_right(Yarr, y0)
    freqG = defaultdict(int)
    freqG[G0] += 1

    # Expand right in y
    cur_y = y0
    cur_G = G0
    cur_cLy = cL0y
    while True:
        nxt_y = cur_y + 1
        nxt_G = cur_G + 2*cur_cLy - N
        if nxt_G > D:
            break
        freqG[nxt_G] += 1
        new_cL = bisect.bisect_right(Yarr, nxt_y)
        cur_y, cur_G, cur_cLy = nxt_y, nxt_G, new_cL

    # Expand left in y
    cur_y = y0
    cur_G = G0
    cur_cLy = cL0y
    while True:
        nxt_y = cur_y - 1
        diff = 2*bisect.bisect_right(Yarr, nxt_y) - N
        nxt_G = cur_G - diff
        if nxt_G > D:
            break
        freqG[nxt_G] += 1
        cur_y, cur_G = nxt_y, nxt_G
        cur_cLy = bisect.bisect_right(Yarr, nxt_y)

    # Now we have freqF and freqG: each is {value_of_F : count_of_x}, {value_of_G : count_of_y}.
    # We want sum_{sx+sy <= D} freqF[sx]*freqG[sy].
    # We'll do a two-pointer approach after sorting the keys.

    Fkeys = sorted(freqF.keys())
    Gkeys = sorted(freqG.keys())

    # Build prefix sums of freqG to quickly sum counts up to an index
    # Then do a standard two-pointer: i from smallest Fkeys, j from largest Gkeys
    # while sx + sy > D => move j down, else accumulate.
    # But we need freqG up to that sy. We'll do it differently:
    # We can do a "cumulative counts" over Gkeys in ascending order, so for each possible sy we know how many y-values sum up to that sy or less.

    # First, convert Gkeys -> (gk, freqG[gk]) in ascending order
    g_list = [(gk, freqG[gk]) for gk in Gkeys]
    # Build a prefix of counts:
    # prefix_count[i] = sum of freq for g_list[:i]
    # prefix_gval[i] = g_list[i-1][0] if needed.  Actually we'll do a 2-pointer approach in ascending order for sy.
    # We'll just do the typical "two pointer" by ascending sx, ascending sy, but we want sy <= D - sx.
    # We'll keep a running sum of freqG up to the pointer.

    # Convert g_list into prefix of the cumulative frequencies for easy sum
    g_prefix_counts = [0]*(len(g_list)+1)
    for i,(gv,fr) in enumerate(g_list):
        g_prefix_counts[i+1] = g_prefix_counts[i] + fr

    ans = 0
    # We'll have an index j that moves from the right to the left to maintain g_list[j].value <= D - sx
    # Actually it's simpler to do an ascending approach: let j move forward while g_list[j].val <= limit
    # We'll do i in ascending order, for each Fkeys[i], define limit = D - Fkeys[i].
    # Then we move j forward while g_list[j].val <= limit. We keep track of j in a pointer that never moves backward overall.

    j = 0
    lenG = len(g_list)
    # g_list is sorted ascending by gk
    for sx in Fkeys:
        frx = freqF[sx]
        limit = D - sx
        # move j forward while j < lenG and g_list[j].val <= limit
        while j < lenG and g_list[j][0] <= limit:
            j += 1
        # all indices < j have g_list[idx].val <= limit
        # so the number of valid y is sum of freqG for idx < j
        count_valid_g = g_prefix_counts[j]
        ans += frx * count_valid_g

    print(ans)