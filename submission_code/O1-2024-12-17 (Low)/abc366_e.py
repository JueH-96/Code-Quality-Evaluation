def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, D = map(int, input_data[:2])
    coords = list(zip(*[map(int, input_data[2:])]*2))
    # coords = [(x1,y1), (x2,y2), ..., (xN,yN)]
    
    # We want all integer (x,y) such that sum_{i=1..N} |x - x_i| + |y - y_i| <= D.
    # Equivalently, define:
    #   Sx(x) = sum_{i=1..N} |x - x_i|
    #   Sy(y) = sum_{i=1..N} |y - y_i|
    # We need Sx(x) + Sy(y) <= D.
    #
    # Plan:
    # 1) Preprocess x_i's and y_i's (sort them, build prefix sums) so we can compute Sx(x) and Sy(y) quickly.
    # 2) Find the feasible integer range [xL..xR] in x for which Sx(x) <= D.  Outside this range, Sx(x) > D,
    #    so no (y) can satisfy the sum <= D.
    # 3) Iterate x from xL to xR (in increasing order).  For each x, let T = D - Sx(x).  Then we need Sy(y) <= T.
    #    Find the integer interval [yL..yR] of y for which Sy(y) <= T.  Count how many integers are in [yL..yR].
    # 4) Sum up over all x.  That gives the total count. 
    #
    # The main challenge is to do this efficiently in Python:
    #   - N can be up to 2*10^5, D up to 10^6.
    #   - The range [xL..xR] can be up to about 2D+1 in the worst case (if all x_i are the same).
    #   - For each x we do a binary search on y to find yL, yR in O(log N), leading to about O(D log N) which can be ~10^6 * 18 = 1.8e7 in the worst case—borderline but can often pass with efficient IO/implementation.
    #
    # Implementation details:
    #   - We'll compute Sx(x) in O(1) as we sweep x from left to right, by keeping track of how many x_i lie to the left/right,
    #     and the sum of those x_i to the left/right.  Then Sx(x+1) = Sx(x) + (left_count - right_count).
    #   - For Sy(y) we keep it in prefix arrays so we can do binary search to find the boundaries.  Each boundary is found in O(log N).
    #
    # Let’s implement.

    sys.setrecursionlimit(10**7)

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    xs.sort()
    ys.sort()

    # Build prefix sums for x and y to allow fast computation of Sy(y) via binary search later.
    # prefix_x[i] = sum of xs[0] + xs[1] + ... + xs[i-1],  prefix_x[0] = 0
    # so sum of xs[l..r-1] = prefix_x[r] - prefix_x[l]
    prefix_x = [0]*(N+1)
    for i in range(N):
        prefix_x[i+1] = prefix_x[i] + xs[i]
    prefix_y = [0]*(N+1)
    for i in range(N):
        prefix_y[i+1] = prefix_y[i] + ys[i]

    # Function Sy(y): sum of |y - ys[i]| for i=0..N-1.
    # We'll do a binary search to find how many of ys are <= y.
    import bisect

    def Sy(y):
        # pos = number of elements in ys that are <= y
        pos = bisect.bisect_right(ys, y)
        # left side: those <= y
        # sum of |y - ys[i]| for i in [0..pos-1] is pos*y - sum_of_that_block
        left_sum = pos*y - (prefix_y[pos])
        # right side: those > y
        right_count = N - pos
        # sum of |y - ys[i]| for i in [pos..N-1] is sum_of_that_block - y*(that_count)
        right_sum = (prefix_y[N] - prefix_y[pos]) - y*right_count
        return left_sum + right_sum
    
    # We want to find the minimal x such that Sx(x) <= D and maximal x.  But we won't recompute Sx(x) naively.
    # Instead, we'll build a data structure for Sx-sweeping in O(1) steps.

    # First, let’s find median-related values to start from a certain x0 where we can compute Sx(x0) easily.
    # A convenient choice is to start from xs[0] or the median.  We'll just do it from xs[0] or any x in that region.
    #
    # Actually, let's pick x0 = xs[0]. Then we can compute Sx(x0) = sum_{i=0..N-1} (xs[0] - xs[i]) if we do it carefully.
    # But that might be big if x0 is not near the median. A more stable approach is to pick x0 as the smallest x_i among the points
    # or just pick x0 = xs[0]. We'll get Sx(x0) in O(N) once, then do the sweep in ascending x from there.
    #
    # However, we actually need to find xL (the left-most integer x with Sx(x) <= D). We can keep stepping left from the smallest x_i
    # but that might go on up to D steps if N=1.  That’s not too large (up to 10^6).  Then we do the same to the right side (another 10^6).
    # We can do that, but we have to be sure we do it in O(1) per step so total up to O(D) which is feasible.

    # Let's build a frequency map for all xs so we can update left_count,right_count easily if we step x by 1.
    from collections import Counter
    freqX = Counter(xs)
    # We also keep left_count = 0, right_count = N initially if we start from something < xs[0].
    # But let's directly start from x0 = xs[0]. We'll compute Sx(x0) in O(N).
    # Then stepping from x to x-1 or x+1 we can update Sx in O(1).

    # But we must find xL by going left until Sx(x) > D, then step back one. Then similarly find xR going right.

    # First, compute Sx(x0) at x0 = xs[0].
    x0 = xs[0]
    # Sx(x0) = sum(|x0 - x_i|) = sum(x_i - x0) for x_i >= x0, or x0 - x_i for x_i <= x0.
    # But here x0 = xs[0], so all x_i >= x0.  So Sx(x0) = sum_{i=0..N-1} (xs[i] - x0).
    Sx_current = 0
    for val in xs:
        Sx_current += (val - x0)

    # We'll maintain:
    #   left_count = number of x_i <= current x
    #   right_count = number of x_i > current x
    #   sum_left = sum of x_i <= current x
    #   sum_right = sum of x_i > current x
    #
    # Sx(current_x) = (current_x * left_count - sum_left) + (sum_right - current_x * right_count).
    # We'll keep these consistent as we move left or right by 1.

    # Build them for x0:
    left_count = 0
    sum_left = 0
    right_count = 0
    sum_right = 0
    # Actually, for x0 = xs[0], all x_i >= x0, so left_count will be how many x_i == x0
    # and right_count the rest.

    # Let’s split the array around x0:
    from bisect import bisect_left, bisect_right
    iL = bisect_left(xs, x0)  # first idx where x >= x0  (this should be 0)
    iR = bisect_right(xs, x0) # first idx where x > x0
    # the x_i == x0 are in [iL..iR-1]
    # left side strictly < x0 is none
    # right side strictly > x0 is [iR..end]

    # count how many are == x0
    eq_count = iR - iL
    left_count = eq_count
    sum_left = x0 * eq_count
    right_count = N - eq_count
    sum_right = sum(xs[iR:])

    # Sx_current as computed above:
    # check with formula:
    #   formula_sx0 = (x0*left_count - sum_left) + (sum_right - x0*right_count)
    # that should match Sx_current
    # but we already computed Sx_current by direct summation. Let's just ensure consistency:
    formula_sx0 = (x0*left_count - sum_left) + (sum_right - x0*right_count)
    # They should match.  If not, we reassign to keep them consistent.
    if formula_sx0 != Sx_current:
        Sx_current = formula_sx0

    # A helper to move from x to x-1 in O(1):
    def move_left():
        nonlocal current_x, Sx_current, left_count, sum_left, right_count, sum_right
        new_x = current_x - 1
        # The difference in Sx is left_count - right_count (when going from x to x-1).
        Sx_current += left_count - right_count
        # Now we need to update left_count, sum_left, right_count, sum_right
        # Because some x_i that were on the right might become on the left if x_i == new_x.
        c = freqX[new_x] if new_x in freqX else 0
        # Those c points now move from the right side to the left side
        left_count += c
        sum_left += new_x * c
        right_count -= c
        sum_right -= new_x * c
        current_x = new_x

    # Similarly move_right():
    def move_right():
        nonlocal current_x, Sx_current, left_count, sum_left, right_count, sum_right
        new_x = current_x + 1
        # The difference in Sx is - (left_count - right_count) (when going from x to x+1).
        Sx_current -= left_count - right_count
        # Now update the counts
        c = freqX[new_x] if new_x in freqX else 0
        left_count -= 0  # see below
        # Actually, those c points were in the left side if they were <= current_x,
        # moving to "equal to new_x" which is now "<= new_x", so they remain left side.
        # Wait, we must be consistent with how we define "left" vs "right": left means x_i <= current_x.
        # After we move to new_x, "left" becomes x_i <= new_x. So any x_i == new_x were previously in the right set (since x_i > current_x).
        # So we must move them from the right set to the left set.
        left_count += c
        sum_left += new_x*c
        right_count -= c
        sum_right -= new_x*c
        current_x = new_x

    # Now find xL by going left while Sx_current <= D. We might need to keep going left until Sx_current > D, then step back.
    current_x = x0
    while True:
        # try moving left once
        move_left()
        if Sx_current > D:
            # revert
            move_right()
            break
    xL = current_x

    # Then find xR by going right while Sx_current <= D
    while True:
        move_right()
        if Sx_current > D:
            # revert
            move_left()
            break
    xR = current_x

    # Now we have the extremal feasible x's for which Sx(x) <= D.  The feasible x are in [xL..xR].
    # Next, we will sweep x from xL to xR in ascending order, computing Sx(x) with our move_right() logic,
    # and for each x, do T = D - Sx(x).  Then find how many integer y satisfy Sy(y) <= T.  We do that by
    # finding the minimal y and maximal y with Sy(y) <= T (using binary search).

    # But first, we want to identify the overall y-range that could possibly matter.  If T >= 0, we only need y s.t. Sy(y) <= T <= D.
    # Let's find the minimal y that can achieve Sy(y) <= D, call it global_yL, and the maximal y that can achieve Sy(y) <= D, call it global_yR.
    # We do that similarly by a left-sweep / right-sweep in the y dimension.  (Or we can do a quick bisection approach, but let's just do
    # a direct method like we did for x, reusing the same patterns.)

    # Build freqY:
    freqY = Counter(ys)

    # We'll do a function that, given a starting y0, returns (yMinFeasible, yMaxFeasible) for Sy(y) <= D.
    # Implementation is exactly like for x.

    def find_feasible_y_range(D):
        # start from y0 = ys[0]
        y0 = ys[0]
        Sy0 = 0
        for val in ys:
            Sy0 += (val - y0)
        # build left_count, sum_left, right_count, sum_right
        lc = 0
        sl = 0
        rc = 0
        sr = 0
        iL = bisect_left(ys, y0)
        iR = bisect_right(ys, y0)
        eqc = iR - iL
        lc = eqc
        sl = y0*eqc
        rc = N - eqc
        sr = sum(ys[iR:])
        Sy_current = Sy0
        # check formula
        formula_sy0 = (y0*lc - sl) + (sr - y0*rc)
        if formula_sy0 != Sy_current:
            Sy_current = formula_sy0
        cur_y = y0

        def move_down():
            nonlocal cur_y, Sy_current, lc, sl, rc, sr
            ny = cur_y - 1
            Sy_current += lc - rc
            c = freqY[ny] if ny in freqY else 0
            lc += c
            sl += ny*c
            rc -= c
            sr -= ny*c
            cur_y = ny

        def move_up():
            nonlocal cur_y, Sy_current, lc, sl, rc, sr
            ny = cur_y + 1
            Sy_current -= lc - rc
            c = freqY[ny] if ny in freqY else 0
            lc += c
            sl += ny*c
            rc -= c
            sr -= ny*c
            cur_y = ny

        # find yMin
        while True:
            move_down()
            if Sy_current > D:
                # revert
                move_up()
                break
        yMin = cur_y
        # find yMax
        while True:
            move_up()
            if Sy_current > D:
                move_down()
                break
        yMax = cur_y
        return (yMin, yMax)

    glob_yL, glob_yR = find_feasible_y_range(D)

    # We also need a fast way to do binary search for Sy(y) <= T.  We'll do:
    #   - find the smallest y in [glob_yL..glob_yR] such that Sy(y) <= T  => call it leftY
    #   - find the largest y in [glob_yL..glob_yR] such that Sy(y) <= T   => call it rightY
    # Then the count is max(0, rightY - leftY + 1).
    # If no y in that range satisfies Sy(y) <= T, the count is 0.

    # We'll define a function is_ok(y, T): return Sy(y) <= T
    # Then we do a binary search in [glob_yL..glob_yR].

    def is_ok(y, T):
        return Sy(y) <= T

    # Standard binary search to find left boundary
    def find_left_boundary(T):
        # we want the minimum y in [glob_yL..glob_yR] with Sy(y) <= T, or if none, return None
        low = glob_yL
        high = glob_yR
        ans = None
        while low <= high:
            mid = (low + high)//2
            if is_ok(mid, T):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        return ans

    # right boundary
    def find_right_boundary(T):
        low = glob_yL
        high = glob_yR
        ans = None
        while low <= high:
            mid = (low + high)//2
            if is_ok(mid, T):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans

    # Now let's do the main sweep over x from xL..xR in ascending order.  But we built everything going from x0 outward.
    # We actually found xL < x0 < xR typically. So we’ll do:
    #   1) jump current_x back down to xL by repeatedly moving_left (since we are at xR now).
    #   2) then sweep up to xR.

    # The easiest might be: we know we ended at xR after the search. Let’s re-initialize from scratch so we can sweep from xL to xR in ascending order.

    # Re-initialize the data structure at xL:
    # We'll do exactly what we did at x0 but now for xL directly in O(N), then build up left_count etc. carefully.

    # We'll gather all x_i in a sorted array xs. xL is an integer <= xs[0], or possibly bigger. We can do a direct sum for Sx(xL).
    # Then we partition around xL for left_count, sum_left, right_count, sum_right.  Then we sweep up.

    # Actually xL might be bigger or smaller than xs[0]. We'll do a direct method:

    def build_x_struct(start_x):
        # compute Sx(start_x) in O(N)
        ssum = 0
        for v in xs:
            ssum += abs(v - start_x)
        # build left_count, sum_left, right_count, sum_right
        # left_count = number of v <= start_x
        # sum_left = sum of those v
        # right_count = number of v > start_x
        # sum_right = sum of those v
        lcount = bisect_right(xs, start_x)  # all v <= start_x
        sl = prefix_x[lcount]
        rcount = N - lcount
        sr = prefix_x[N] - sl
        return ssum, lcount, sl, rcount, sr

    current_x = xL
    Sx_current, left_count, sum_left, right_count, sum_right = build_x_struct(xL)

    answer = 0
    move_right_once = (freqX[current_x+1] if (current_x+1) in freqX else 0)  # prefetch? We'll just define a function.

    x_end = xR

    while True:
        # Now T = D - Sx_current
        T = D - Sx_current
        if T >= 0:
            # find how many y in [glob_yL..glob_yR] satisfy Sy(y) <= T
            leftY = find_left_boundary(T)
            if leftY is not None:
                rightY = find_right_boundary(T)
                # number of integer y in [leftY..rightY]
                answer += (rightY - leftY + 1)
        if current_x == x_end:
            break
        # move to current_x+1
        new_x = current_x + 1
        # update Sx in O(1)
        Sx_current -= left_count - right_count
        c = freqX[new_x] if new_x in freqX else 0
        left_count += c
        sum_left += new_x*c
        right_count -= c
        sum_right -= new_x*c
        current_x = new_x

    print(answer)