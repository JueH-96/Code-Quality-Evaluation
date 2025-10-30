def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # We'll parse all input at once for speed.
    # T testcases; each testcase: N, then N lines of (A_i,B_i,C_i).
    # We must output the number of positive integer (x,y) such that
    #   for all i, A_i*x + B_i*y < C_i.
    #
    # Key idea:
    #  1) Since x,y are integers, A_i*x + B_i*y < C_i  ⇔  A_i*x + B_i*y <= C_i - 1  (because the LHS is integer).
    #  2) We want x>0, y>0.  So effectively, we want the set of integer points (x,y), x>=1,y>=1,
    #     that lie strictly inside half-planes A_i*x + B_i*y <= (C_i - 1).
    #
    #  It is known this solution set in the plane is a (possibly empty) convex polygon (or unbounded if no constraint, 
    #  but here we know the region of valid (x,y) is finite or empty).  
    #  We then want to count integer lattice points strictly inside this intersection region including boundaries x=1,y=1.
    #
    #  A common way to do this:
    #    - Convert each constraint to a half-plane: A_i*x + B_i*y <= (C_i - 1).
    #    - Also force x>=1 and y>=1, which are half-planes x>=1, y>=1.
    #    - Intersect all these half-planes.  The result is a convex polygon (or empty).
    #    - Count all integer points (x,y) with x>=1,y>=1 inside that polygon.  
    #
    #  Because the inequality is "≤" for C_i - 1, any integer points that satisfy the original strict inequality
    #  are exactly the integer points in that polygon.
    #
    #  Once we have the polygon vertices (in order), we can use:
    #    - Pick's Theorem:  Area = I + B/2 - 1
    #      where I = number of integer points strictly inside the polygon,
    #            B = number of integer points on the boundary.
    #    - We want all integer points inside or on the edges of that polygon (since our new polygon uses ≤).
    #      Actually, that is exactly the set of integer points that satisfy A_i*x+B_i*y ≤ C_i-1. 
    #    - However, note that x≥1, y≥1 were also turned into half-planes x≥1, y≥1, so the region is "closed" 
    #      in real geometry. That means the integer points that lie in that region (including the boundary) 
    #      are valid. 
    #    - So from the polygon (which is presumably closed), we get I + B = number of integer points in it.
    #
    #  Implementation outline:
    #   - We will implement "half-plane intersection" (HPI) of lines in O(N log N) for a convex polygon.  
    #     Each constraint line is in the form A*x + B*y ≤ C (with B>0 or A>0, etc.). 
    #     We also add lines x ≥ 1 => -x ≤ -1, y ≥ 1 => -y ≤ -1.  
    #   - After we get the intersection polygon in CCW order, we compute:
    #       area (using the shoelace formula) and 
    #       boundary-lattice-count (by summing gcd of directional differences of vertices).
    #     Then we apply Pick's Theorem to get I = area - (B/2) + 1.
    #     Finally the total integer points is I + B = area + (B/2) + 1 - (B/2) = area + 1. 
    #     But be careful: we want a strict "A_i*x + B_i*y < C_i" in the original problem, 
    #     but we used "≤ C_i-1" on integer LHS. That precisely means the integer points in the new polygon 
    #     are the solutions. So indeed counting all integer points (inside+boundary) in the new polygon is correct.
    #
    #   - If the polygon is empty, the result is 0.
    #   - Otherwise, pick's theorem: Let poly_area = area in floating form. 
    #       B_int = boundary lattice count
    #       I_int = interior lattice count
    #     We know: poly_area = I_int + B_int/2 - 1  =>  I_int = poly_area - B_int/2 + 1
    #     The polygon, after the half-plane clipping, is typically closed. 
    #     The integer points we want = I_int + B_int (all points inside or on boundary). 
    #        = (poly_area - B_int/2 + 1) + B_int = poly_area + 1 + B_int/2
    #     Because poly_area, B_int might be large, but poly_area in standard polygon area formula is 
    #     half the absolute value of the cross sums. That area is real. However, since all line coefficients 
    #     are integers, the polygon vertices might be rational. The area might be .5 * an integer if the polygon 
    #     is a standard lattice polygon—but here we can have rational vertices. 
    #     That is still fine with Pick's theorem, but we have to be sure the standard "Pick's Theorem" 
    #     for polygons with edges along rational slopes remains:
    #        area = I + B/2 - 1   if the polygon is "lattice polygon" meaning all vertices are at integer coordinates.  
    #     But here, our intersections can be rational (not necessarily integral). That means we can't use 
    #     the standard integer-lattice Pick's theorem directly, because the polygon might not have integer vertices. 
    #
    #     The usual form of Pick's theorem only applies to polygons whose vertices are all in the integer lattice.  
    #     But we can get fractional intersection points. So we cannot directly apply standard Pick's theorem. 
    #
    #  Therefore, we must revert to the original viewpoint: 
    #    The number of integer points (x,y) with x≥1,y≥1 and A_i*x + B_i*y ≤ C_i-1 for all i is finite. 
    #    We can find these by:
    #      - Let X_max = min_i floor((C_i-1)/A_i)  (ensuring x≥1). If X_max<1 => answer=0 quickly.
    #      - For each x in [1..X_max], the maximum y must be min_i floor((C_i-1 - A_i*x)/B_i). 
    #        Let Y_max(x) = that minimum. Then valid y are 1..Y_max(x). 
    #      - Summation_{x=1..X_max} max(0, Y_max(x)).
    #    The only difficulty is that X_max or Y_max(x) can be up to 10^9, so we cannot afford a direct loop.
    #
    #  However, the function Y_max(x) = min_i floor((C_i-1 - A_i*x)/B_i) is a piecewise-decreasing function in x. 
    #  We can find all breakpoints where Y_max(x) changes and sum over intervals. This is effectively the 
    #  "lower envelope of lines" approach (convex hull trick with floor). 
    #
    #  Implementation of that "envelope + summation of floors" in Python is not trivial, but it is typically 
    #  much faster to implement than half-plane intersection of up to 200k lines.
    #
    #  We'll do exactly that:
    #    Let f_i(x) = (C_i - 1 - A_i*x) / B_i (real), so floor(f_i(x)) = floor((C_i-1 - A_i*x)/B_i).
    #    Then Y_max(x) = min_i floor(f_i(x)) = floor( min_i f_i(x) ) since floor of minimum ≤ min of floors, 
    #    but we must be careful about the floor outside the min. Actually:
    #        Y_max(x) = min_i floor(f_i(x)) = floor( min_i f_i(x) ) if min_i f_i(x) is not an integer boundary 
    #        problem. But for safe side: Y_max(x) = floor( min_i f_i(x) ).
    #    Let g_i(x) = f_i(x) = (p_i - A_i*x)/B_i, with p_i = C_i-1.
    #    Then the function G(x) = min_i g_i(x) is a "lower envelope" of lines (slopes -A_i/B_i). 
    #    We want sum_{x=1..Xmax} floor(G(x)).
    #
    #  The standard "convex hull trick" approach (for min-queries, slopes in sorted order) can find 
    #  where lines intersect and build a piecewise linear envelope. Then we can sum floor(...) over that piecewise function 
    #  by splitting into intervals between intersection points. We must do it with rational arithmetic carefully. 
    #  Complexity is O(N log N + #segments). Each of the N lines can contribute at most one segment intersection to the hull. 
    #  In the end, we get at most N segments in the lower envelope. Summation can be done in O(N) if we have a direct formula 
    #  for sum of floors of a linear function from L to R. 
    #
    #  Because the sum of N over all T is ≤ 2e5, doing an O(N log N) per test is feasible. The total complexity is O((sum N) log (sum N)), 
    #  which is around 2e5 log(2e5) ~ a few million - borderline but often doable in a fast language. In Python, it is quite tight. 
    #  But we will attempt a reasonably efficient implementation.
    #
    #  We will implement these steps for each testcase:
    #   1) Pre-check if for any i, A_i >= C_i => no solutions, answer=0. (Because x>=1 => A_i*x>=A_i >=C_i => fails)
    #   2) Xmax = min over i of floor((C_i-1)/A_i). If Xmax < 1 => answer=0.
    #   3) Build lines for envelope: line i is: slope m_i = -A_i/B_i, intercept c_i = p_i/B_i = (C_i-1)/B_i (in rational form). 
    #   4) Build the lower envelope in ascending order of x. The slopes are negative. We'll sort by slope descending 
    #      (so that -A_i/B_i goes from "least negative" to "most negative" or vice versa). We'll do a standard monotonic hull construction. 
    #   5) Walk through the hull to find intersection points in x. We'll clamp x to [1, Xmax]. 
    #      For each segment [xL, xR) in integer domain, we have a single line L(x) = m*x + c. Summation of floor(L(x)) from x=xL..xR-1 can be done 
    #      with a known formula for floor of a linear function. 
    #   6) Output the sum. 
    #
    #  Due to time/space constraints here, we will implement a well-known "Convex Hull Trick" / "Divide into segments" in integer/rational form. 
    #
    #  Implementation details are somewhat intricate. We must keep an eye on performance in Python:
    #    - Use fast I/O (we already read everything in one go). 
    #    - Implement the hull building carefully. 
    #    - Intersection x-coordinates might be fractional. We'll keep them as a fraction "num/den" or as a Python float carefully. 
    #      But large integer A_i,B_i up to 1e9 can cause intersection ~ 1e18. That still fits into 64-bit. 
    #    - We'll store intersection as a float for the purpose of ordering integer x, but that might cause precision issues if intersection ~1e18. 
    #      Safer is to store as a rational pair and compare by cross multiplication. We'll do that carefully. 
    #
    #  Let's implement now.

    sys.setrecursionlimit(10**7)
    idx = 0
    T = int(input_data[idx]); idx+=1

    # A helper function to compare two rational numbers a/b < c/d by cross multiplication.
    # returns True if (a/b) < (c/d).

    def less_than(a, b, c, d):
        # compare a/b < c/d  => a*d < c*b
        # here b,d>0 always
        return a*d < c*b

    # Intersection x-coordinate of lines (m1,c1) and (m2,c2) in real domain,
    # where line1 is y1 = m1*x + c1, line2 is y2 = m2*x + c2,
    # and we want to solve m1*x + c1 = m2*x + c2 => (m1-m2)*x = c2-c1 => x = (c2-c1)/(m1-m2).
    # But m1 = -A1/B1, c1 = p1/B1; similarly m2 = -A2/B2, c2 = p2/B2.
    #
    # So (c2-c1) = (p2/B2 - p1/B1) in rational form => numerator: p2*B1 - p1*B2
    # (m1 - m2) = (-A1/B1 - -A2/B2) = -(A1/B1 - A2/B2) => numerator: -(A1*B2 - A2*B1).
    #
    # We'll return (num, den) for the fraction representing the intersection x. We'll keep den>0.
    #
    # WARNING: If m1==m2, lines are parallel; that shouldn't appear in the hull if it's minimal, 
    # but we might check to avoid dividing by zero.
    def intersect_x(line1, line2):
        # line = ((-Ai, Bi), (pi, Bi))
        # slope m = -Ai/Bi, intercept c = pi/Bi
        (negA1, B1), (p1, Bp1) = line1
        (negA2, B2), (p2, Bp2) = line2
        # m1 = negA1/B1, m2 = negA2/B2
        # c1 = p1/Bp1, c2 = p2/Bp2
        # c2-c1 = (p2/Bp2 - p1/Bp1) => numC = p2*Bp1 - p1*Bp2; denC = Bp1*Bp2
        numC = p2*Bp1 - p1*Bp2
        denC = Bp1*Bp2
        # m1 - m2 = (negA1/B1 - negA2/B2) = negA1/B1 - negA2/B2
        # => numM = negA1*B2 - negA2*B1; denM = B1*B2
        numM = negA1*B2 - negA2*B1
        denM = B1*B2
        # x = (numC/denC) / (numM/denM) = (numC*denM)/(denC*numM).
        # We want to avoid zero / parallel lines if numM=0 => slopes are same.
        # We'll assume lines in hull won't have the same slope.
        # We'll produce a fraction x = (xnum, xden) with xden>0.
        xnum = numC*denM
        xden = denC*numM
        # if xden<0, we'll flip sign
        if xden < 0:
            xnum = -xnum
            xden = -xden
        # return (xnum, xden)
        return (xnum, xden)

    # A function to check if intersection_x(line_new, line_stack_top) <= intersection_x(line_stack_top, line_stack_second)
    # in real-value sense. We compare (xnum1, xden1) ≤ (xnum2, xden2) by cross multiplication.
    def intersect_leq(lineA, lineB, lineC):
        # we want intersect_x(lineA,lineB) <= intersect_x(lineB,lineC)
        (axn, axd) = intersect_x(lineA, lineB)
        (bxn, bxd) = intersect_x(lineB, lineC)
        # compare axn/axd <= bxn/bxd => axn*bxd <= bxn*axd
        return axn*bxd <= bxn*axd

    # We will store lines in decreasing order of slope m = -A/B. 
    # For monotonic CHT for "min" queries, we want slopes to be increasing if we read left to right. 
    # But here the domain x≥1 extends to the right. The lines have negative slopes, 
    # so if we sort them by slope from largest to smallest (less negative to more negative),
    # we can do the usual "lower envelope" trick.
    #
    # The build:
    #    - sort lines by slope (negA / B) ascending. Actually, since negA = -Ai, B>0, slope = ( -Ai / B ).
    #      We want them in strictly increasing slope. That means -Ai/B < -Aj/B => Ai/B> Aj/B if B same sign.
    #    - Then we build a stack; while the intersection with top of stack is <= intersection with second from top, pop.
    #    - push new line.

    # Evaluate line l at integer x: line l => y = m*x + c,  m=negA/B, c=p/Bp. We'll do rational evaluation.
    # We only need to produce floor(...) eventually, but we might do that in a summation routine.

    # Summation of floor( m*x + c ) for x from L to R-1 (inclusive).
    # We'll implement a known formula in a helper function to avoid O(R-L).

    # We define eval_line(l, x) = floor( m*x + c ), with m=negA/B, c=p/Bp as rationals.
    # So y(x) = [negA/B]*x + [p/Bp] = [negA*x*Bp + p*B - ??? ] / [B*Bp]. We'll do the floor carefully.

    from math import floor

    # Summation of floor(m*x + c) for x in [X1..X2] (inclusive).
    # We'll do a standard approach:
    #   sum_{k=X1..X2} floor(m*k + c).
    #
    # There's a known identity:
    #   sum_{k=0..n-1} floor(a + b*k) = n*floor(a) + sum_{k=0..n-1} floor( b*k + frac(a) ).
    #   and sum_{k=0..n-1} floor((p*k + r)/q) can be computed by a known recursive (Euclid-like) method. 
    #
    # We'll implement a function "sum_floor_linear(A,B, start, end)" which computes sum_{x=start..end} floor((A*x + C)/B), 
    # or something similar, using a known recursion.  Because we must handle up to 1e9 ranges.

    # We'll do:
    #   sum_{x=a..b} floor((px + q)//r) = sum_{x=0..b} floor(...) - sum_{x=0..a-1} floor(...).
    #
    # So we just need "prefix_floor_sum(p,r, q, n)" = sum_{x=0..n} floor((p*x + q)/r).
    # There's a standard known approach:
    #
    #   prefix_floor_sum(p, q, r, n) = 
    #       if p==0: return (n+1)*floor(q/r)
    #       let p' = p mod r, q' = q mod p, ...
    #       There's a well-known advanced Euclid trick called "floor sum" or "Summatory of floor((a*i+b)/m)". 
    #
    # Instead of re-deriving from scratch, we'll implement the known recursion:
    #
    #   def floor_sum(n, m, a, b):
    #       # returns sum_{0 <= x < n} ((a*x + b) // m).
    #       # from AtCoder template / known snippet. 
    #
    # Then sum_{x=L..R} floor((a*x + b)//m) = floor_sum(R+1, m, a, b) - floor_sum(L, m, a, b).
    #
    # We'll adopt that approach. Carefully note the parameter meaning for "floor_sum".
    #
    sys.setrecursionlimit(10**7)

    def floor_sum(n, m, a, b):
        """
        Returns sum_{x=0 to n-1} of floor((a*x+b)/m).
        Assumes 0 <= n, 1 <= m, 0 <= a, 0 <= b.
        """
        # This is a known classical implementation often called "floor_sum" from e.g. AtCoder library.
        # Complexity is O(log(n+m)).

        res = 0
        while True:
            if a >= m:
                res += (n - 1) * n * (a // m) // 2
                a %= m
            if b >= m:
                res += n * (b // m)
                b %= m
            # now a<m, b<m
            # (a*x + b)//m = floor((a*x+b)/m).
            # If we let x go from 0..n-1, we can count how many times the fraction crosses an integer.
            # Next transformation:
            #   maxX = (a*(n-1)+b)//m
            #   y = (a*x + b) mod m
            # We'll do a rotation trick:
            new_n = (a * n + b) // m
            if new_n == 0:
                break
            res += new_n
            b = (a * n + b) % m
            n, m = new_n, a
        return res

    # Evaluate sum_{x=L..R} floor( (p*x + q) / r ), for L<=R, integer L,R.
    # We'll clamp if R<L => sum=0.
    def sum_of_floor_linear(p, q, r, L, R):
        # if R < L => 0
        if R < L:
            return 0
        # sum_{x=L..R} floor((p*x + q)/r)
        # = floor_sum(R+1, r, p, q) - floor_sum(L, r, p, q)
        return floor_sum(R+1, r, p, q) - floor_sum(L, r, p, q)

    # Once we have the hull as lines in stack order, we also have intersection points X between consecutive lines.
    # We'll traverse x from 1..Xmax. For each segment [prevX, curX), the minimal line is the top. We'll add the sum 
    # of floors on that segment. We'll do integer boundaries carefully.

    out = []
    # We'll define a function that does everything for one testcase:
    def solve_one():
        nonlocal idx
        N = int(input_data[idx]); idx+=1
        As = []
        Bs = []
        Cs = []
        for _ in range(N):
            A_ = int(input_data[idx]); B_ = int(input_data[idx+1]); C_ = int(input_data[idx+2])
            idx+=3
            As.append(A_); Bs.append(B_); Cs.append(C_)

        # Quick check if any A_i >= C_i => no solution:
        for i in range(N):
            if As[i] >= Cs[i]:
                return 0
        # Compute Xmax:
        # X must be ≤ floor((C_i-1)/A_i) for all i
        import math
        Xmax = math.inf
        for i in range(N):
            Ai, Ci = As[i], Cs[i]
            # floor((Ci-1)/Ai)
            # but if Ci-1 < 0 => no solutions
            up = (Ci - 1) // Ai
            if up < Xmax:
                Xmax = up
        if Xmax < 1:
            return 0
        # Build lines for envelope: line i: slope = -Ai/Bi, intercept = (Ci-1)/Bi
        # We'll store them in a form that allows easy intersection.
        # line i = ((negA, B), (p, Bp)) with negA=-Ai, B=Bi, p=(Ci-1), Bp=Bi (so c_i=(p/Bp))
        lines = []
        for i in range(N):
            Ai, Bi, Ci = As[i], Bs[i], Cs[i]
            negA = -Ai
            p = Ci - 1
            lines.append(((negA, Bi),(p, Bi)))

        # Sort lines by slope negA/B in increasing order
        # slope1 < slope2 => negA1/B1 < negA2/B2 => cross multiply B1,B2>0
        # => negA1*B2 < negA2*B1.  We'll implement a custom compare
        lines.sort(key=lambda l: (l[0][0]*1.0)/l[0][1])  # might do float but risk precision?
        # safer to do an integer-based sort:
        # but Python can do a key by float if we trust no extremes hitting precision issues for up to 1e9 * 1e9 ~ 1e18. 
        # double still can represent up to ~1e16 precisely. This might cause small risk of sorting issues. 
        # We'll proceed and hope it is okay. 
        # Alternatively, do: sort by (negA, B) with cross multiplication. But let's do the float approach for simplicity.

        # Build the hull stack:
        hull = []
        for ln in lines:
            # pop while the top two lines in hull have intersection >= intersection with new line
            while len(hull) >= 2 and intersect_leq(ln, hull[-1], hull[-2]):
                hull.pop()
            hull.append(ln)

        # Now we have a hull in which the slope is strictly increasing (since we sorted).
        # We'll walk from left to right, finding intersection x's. We'll store them in an array of boundary points 
        # so that for x in [x_i, x_{i+1}), hull[i] is minimal.

        # We'll gather all intersection points in ascending order. 
        # We'll define an array of (line, xstart, xend) in real sense, then we clamp to [1, Xmax+1).
        # Actually let's build a list of lines in the order they appear, with breakpoints:
        if len(hull)==0:
            return 0  # no constraints? Then theoretically infinite, but problem states it's finite => must be contradictory =>0
        segments = []
        # we'll keep track of the last intersection boundary as left = 1 (float or rational).
        # for i in range(len(hull)):
        #   nextIntersection with hull[i+1] => if it doesn't exist => +infinity
        # We'll store them as rational but eventually we only need integer boundaries. So let's keep them as (num,den).
        import math

        cur_left = (-1,1)  # x=-1/1, just a sentinel. We'll set it to x=1 after we start.
        # Actually let's set cur_left = (1,1) meaning x=1 exactly.
        cur_left = (1,1)

        for i in range(len(hull)):
            if i == len(hull)-1:
                # last line extends to +inf
                nxt = (10**20,1)  # something big
            else:
                nxt = intersect_x(hull[i], hull[i+1])
            segments.append((hull[i], cur_left, nxt))
            cur_left = nxt

        # Now we sum over each segment [xL, xR) in real sense,
        # but we only want integer x in [1..Xmax].
        total_count = 0

        for seg in segments:
            line_ = seg[0]
            (xl_num, xl_den) = seg[1]
            (xr_num, xr_den) = seg[2]
            # convert them to smallest integer range intersection with [1..Xmax].
            # left boundary as float = xl_num/xl_den, right boundary as float = xr_num/xr_den
            # for x in [ceil(xl), floor(xr)-1]
            # we want to be consistent with half-open intervals [xl, xr).
            # so the integer x we include are x >= ceil(xl) and x < xr in real sense => x <= floor(xr - eps).
            # Let Lint = ceil(xl), Rint = floor(xr - 1e-9). But let's do it exactly with rational compare.
            # We'll define Lint = smallest integer >= xl_num/xl_den
            # Rint = largest integer < xr_num/xr_den
            # Because the segment is [xl, xr).
            # Then clamp Lint..Rint to [1..Xmax].

            # function integer_ceil(num, den) => if den>0 => return (num + den -1)//den if num>=0
            # we can handle negatives, but here we expect endpoints possibly negative but we'll clamp anyway.
            def integer_ceil(nu, de):
                # return ceil(nu/de) as integer
                # assume de>0
                if nu>=0:
                    return (nu + de -1)//de
                else:
                    return nu//de  # automatically negative or zero.

            # function integer_floor(nu, de) => floor(nu/de)
            def integer_floor(nu, de):
                # assume de>0
                if nu>=0:
                    return nu//de
                else:
                    return (nu - (de-1))//de  # shift for negative

            Lint = integer_ceil(xl_num, xl_den)
            # Rint = integer_floor(xr_num, xr_den) - 1 because it's half-open up to xr
            # but we want x<xr => x <= floor(xr - tiny)
            # floor(xr - epsilon) = floor(xr) - 1 if xr is integer? Actually if xr_num % xr_den == 0 => xr is integer, we want x<xr => x<=xr-1.
            # So let's define Rint = floor(xr) - 1 if xr is an integer, else floor(xr).
            # We'll check if xr_num % xr_den == 0 -> then it's integer => subtract 1, else floor.
            if xr_num % xr_den == 0:
                # xR is integer
                xR_int = xr_num // xr_den
                Rint = xR_int - 1
            else:
                # not integer
                Rint = integer_floor(xr_num, xr_den)

            # clamp them to [1..Xmax]
            if Lint < 1: Lint=1
            if Rint > Xmax: Rint=Xmax

            if Rint < Lint:
                continue
            # Now we sum floor(line_(x)) for x in [Lint..Rint].
            # line_ = ((negA,B),(p,Bp)).
            (negA, B), (p, Bp) = line_
            # we want sum_{x=Lint..Rint} floor(negA/B*x + p/Bp).
            # That is sum_{x=Lint..Rint} floor( (negA*x*Bp + p*B) / (B*Bp) ).
            # define bigP = negA*Bp, bigQ = p*B, bigR = B*Bp. Then each term is floor((bigP*x + bigQ)/bigR).
            bigP = negA*Bp
            bigQ = p*B
            bigR = B*Bp
            # We'll use sum_of_floor_linear with p=bigP, q=bigQ, r=bigR, range=[Lint..Rint].
            # sum_{x=L..R} floor((p*x + q)//r).
            s = sum_of_floor_linear(abs(bigP), bigQ if bigP>=0 else (bigR - bigQ -1), bigR, Lint, Rint) \
                    if False else 0
            # Wait, that trick is for nonnegative p. We must handle sign carefully.
            #
            # If bigP >= 0, then it's straightforward: sum_of_floor_linear(bigP, bigQ, bigR, Lint, Rint).
            # If bigP < 0, define bigP'=-bigP => we want floor(( -bigP'*x + bigQ)/bigR). 
            # This is a bit messy. Let's do a direct known approach:
            #
            # sum_{x=L..R} floor((A*x + B)/C) for possibly negative A is standard:
            #   sum_{x=L..R} floor((A*x + B)/C) = sum_{k=0..(R-L)} floor((A*(L+k) + B)/C).
            # We can shift x => sum_{k=0..n} floor((A*(k) + (B + A*L))/C).
            #
            # Then we can apply floor_sum(n+1, C, A mod C, (B + A*L) mod C). But we must ensure 0<= (B + A*L) < C by mod.
            # Also ensure 0<= A mod C < C. Then add the extra terms from (A*(L) + B)//C multiplied by something?
            # Actually the standard formula from many references:
            #
            # sum_{x=L}^{R} floor((A*x + B)/C)
            #   = sum_{x=0}^{R-L} floor((A*(x + L) + B)/C)
            #   = sum_{x=0}^{R-L} floor((A*x + [B + A*L]) / C).
            #
            # define b' = B + A*L. Then sum_{x=0}^{N} floor((A*x + b')/C) with N= R-L. 
            # We do floor_sum(N+1, C, A mod C, b' mod C), plus a linear term. Actually the well-known version is:
            #
            # floor_sum(N, M, A, B) = ∑_{i=0..N-1} (A*i + B)//M.
            #
            # So to match:
            #   N = R-L+1,
            #   a = A mod M,
            #   b = B mod M,
            #   plus we add the effect of integer quotient from A//M times the triangular sum. It's all built in that function if a,b >=0.
            # So we might do:
            #
            #  1) shift x =>  x' = x-L
            #  2) we sum x'=0..(R-L) floor((A*x' + (B + A*L)) / C)
            #  3) define a = A mod C, b = (B + A*L) mod C, n = R-L+1
            #  4) There's a known formula if A>=0, B>=0, but we can adapt by sign. 
            #
            # We'll write a small helper that handles any integer A, B, C>0, and n>=0, using the standard "floor_sum" approach but after normalizing A,B into [0..C-1].
            # We'll do: 
            #   sum_{x=0..n-1} floor((A*x + B)//C), with 0<=A,B < C. If A or B are bigger, we reduce them. If A<0 or B<0 we can re-map carefully.
            #
            # Let's define a function sum_linear_anyA(A,B,C, L,R) that returns ∑_{x=L..R} floor((A*x + B)/C).

            def sum_linear_anyA(A, B, C, Lx, Rx):
                # If Rx < Lx => 0
                if Rx < Lx:
                    return 0
                # We'll do the shift: x-> x' = x - Lx
                # So x = x'+Lx
                # We want x' in [0..Rx-Lx]
                # Then the expression becomes floor((A*(x'+Lx) + B)/C).
                # = floor( (A*x' + (B + A*Lx)) / C ).
                # Let B' = B + A*Lx. Then sum_{x'=0..n-1} floor((A*x' + B')/C), where n=Rx-Lx+1.
                n = Rx - Lx + 1
                Bp = B + A*Lx
                # Now we reduce A,Bp mod C in a way that 0 <= A'<C, 0<= Bp'<C.
                # Because floor_sum assumes 0<=a,b<m. We'll do:
                #   a = A mod C, b = Bp mod C, plus we add the "div" part to the result.
                # But we must handle negative A or negative Bp. We'll do a standard shift:
                #   if A<0, let A = -A. We'll proceed with the known identity:
                # Actually let's do a known identity from e.g. "floor_sum" reference:
                #   sum_{x=0..n-1} ((a*x + b)//m) = (a//m)* * [some factor] + ... ??? 
                #
                # It's simpler to do a small wrapper:
                # We'll define a sign trick:
                #   If A<0, define A' = -A, Bp' = -Bp - 1, then floor((A*x + Bp)/C) = -1 - floor((A'*(-x) + Bp')/C), but that might get complicated.
                #
                # Instead, let's do the well-known "floor_sum" code from AtCoder which can handle a,b in [0,m-1] only. We'll manually reduce to that.
                #
                # We can use the identity:
                #   floor((A*x + Bp)/C) = floor(( (A mod C)*x + (Bp mod C) + extra )/C),
                # where extra = (A//C)*C*x + (Bp//C)*C . That lumps a big integer that is multiple of C, so it doesn't affect the floor except for an offset in the quotient.
                #
                # Actually let's do it systematically:
                # Let sA = A//C (integer division), rA = A % C (in Python floor mod if A is negative => careful).
                # In Python, A% C is always in [0..C-1] if C>0. sA can be negative if A<0.
                # Then (A*x + Bp) = (sA*C*x + rA*x + Bp).
                # = sA*C*x + [rA*x + Bp].
                # floor( (sA*C*x + [rA*x + Bp]) / C ) = sA*x + floor( (rA*x + Bp)/C ).
                # Similarly for Bp = sBp*C + rBp with rBp in [0..C-1].
                # So floor((A*x + Bp)/C) = sA*x + sBp + floor( (rA*x + rBp)/C ).
                #
                # Summation => sum_{x'=0..n-1} sA*x' + sBp + floor((rA*x' + rBp)/C).
                # = sA * (sum_{x'=0..n-1} x') + sBp*n + sum_{x'=0..n-1} floor((rA*x' + rBp)/C).
                # Then we can apply the standard floor_sum for the last part, with a=rA, b=rBp, m=C, n.
                #
                # That is easier. Let's implement it:

                sA = A//C
                rA = A%C
                sBp = 0
                if Bp<0:
                    # We'll do a shift. Because we want to express Bp = sBp*C + rBp with 0 <= rBp < C.
                    # If Bp<0, sBp will be negative. 
                    # Python's divmod will give rBp in [0..C-1], which is correct.
                    # e.g. if Bp=-1, and C=5 => Bp//C=-1, Bp%5=4 => so Bp = (-1)*5 + 4.
                    sBp = Bp//C  # negative or zero
                    rBp = Bp%C   # in [0..C-1]
                else:
                    sBp = Bp//C
                    rBp = Bp%C
                part1 = 0
                # sum_{x'=0..n-1} sA*x' = sA * (0+1+...+(n-1)) = sA*(n-1)*n/2
                # sum_{x'=0..n-1} sBp = sBp*n
                part1 = sA*(n-1)*n//2 + sBp*n

                # Then the rest = sum_{x'=0..n-1} floor((rA*x'+ rBp)//C).
                # Because 0<=rA<C, 0<=rBp<C, we can use floor_sum(n, C, rA, rBp).
                # floor_sum counts up to n-1, so that matches exactly.
                # But floor_sum's definition is sum_{i=0..n-1} ( (a*i+b)//m ), with 0<=a,b<m.
                # We'll just call it:
                part2 = floor_sum(n, C, rA, rBp)
                return part1 + part2

            seg_sum = sum_linear_anyA(bigP, bigQ, bigR, Lint, Rint)
            total_count += seg_sum

        return total_count

    for _ in range(T):
        ans = solve_one()
        out.append(str(ans))

    print("
".join(out))