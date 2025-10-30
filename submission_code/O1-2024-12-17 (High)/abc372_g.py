def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    
    # We'll parse the input manually (faster than repeated input())
    # and implement the solution for each test case.
    # Index for parsing:
    idx = 1
    
    # A small utility for extended Euclidean floor-sum.
    # This computes: sum_{k=0..n-1} floor((a*k + b) / m).
    # Assumes n >= 0, m > 0, and 0 <= a,b < m on entry (after we reduce them).
    # Reference: "floor_sum" from AtCoder library.
    def floor_sum(n, m, a, b):
        """
        Returns sum_{k=0..n-1} ((a*k + b)//m).
        0 <= a,b < m, n >= 0, m > 0.
        """
        ans = 0
        while True:
            if a >= m or b >= m:
                ans += (n*(n-1)//2)*(a//m)
                ans += n*(b//m)
                a %= m
                b %= m
            # max_x = the largest value so that a*(n-1)+b < m * (max_x+1)
            # or y in the usual code.
            # y = ((a*(n-1)+b)//m)
            y = ((a*(n-1) + b) // m)
            if y == 0:
                return ans
            ans += y
            b = a*(n-1) + b - y*m
            n = y
            # swap(m,a)
            m, a = a, m

    # A function to compute sum_{y=L..R} floor((a*y + b)/c) for arbitrary integer a,b,c.
    # This uses the decomposition:
    #   (a*y + b)//c = (a//c)*y + (b//c) + ((a % c)*y + (b % c))//c
    # and calls the above floor_sum for the remainder part, taking care of offsets.
    def sum_floor_line(a, b, c, L, R):
        """
        Computes ∑_{y=L..R} floor((a*y + b)//c), handling possibly negative a,b.
        """
        if L > R:
            return 0
        # We'll use a helper: calc(N, a, b, c) = sum_{k=0..N-1} floor((a*k + b)/c).
        def calc(N, A, B, C):
            if N <= 0:
                return 0
            # Decompose A,B w.r.t. C
            Adiv, Arem = divmod(A, C)
            Bdiv, Brem = divmod(B, C)
            # Python's divmod can give negative remainders if A < 0, so fix:
            if Arem < 0:
                Arem += C
                Adiv -= 1
            if Brem < 0:
                Brem += C
                Bdiv -= 1
            
            # sum_{k=0..N-1} floor((A*k + B)/C)
            # = sum_{k=0..N-1} [Adiv*k + Bdiv + floor((Arem*k + Brem)/C)]
            # = Bdiv*N + Adiv*(N*(N-1)//2) + sum_{k=0..N-1} floor((Arem*k + Brem)//C).
            part = Bdiv*N + Adiv * (N*(N-1)//2)
            part += floor_sum(N, C, Arem, Brem)
            return part
        
        return calc(R+1, a, b, c) - calc(L, a, b, c)

    # We will build a "Convex Hull Trick" structure for min queries in ascending y.
    # Each line is in the form f(y) = (a*y + b) / c.  We'll store them as (a, b, c).
    # We also store x_intersect which is the boundary from which it becomes better than the previous line.
    # The slopes s_i = a_i / c_i will be sorted in ascending order.

    # Intersection of line1=(a1,b1,c1), line2=(a2,b2,c2) in real y:
    # f1(y)=f2(y) -> (a1*y + b1)/c1 = (a2*y + b2)/c2
    # => (a1*y + b1)*c2 = (a2*y + b2)*c1
    # => a1*c2*y + b1*c2 = a2*c1*y + b2*c1
    # => (a1*c2 - a2*c1)*y = b2*c1 - b1*c2
    # => y = (b2*c1 - b1*c2)/(a1*c2 - a2*c1)
    # We'll store (num, den) with den>0 for the intersection, or handle parallel lines separately.

    def intersect(l1, l2):
        (a1, b1, c1) = l1
        (a2, b2, c2) = l2
        num = b2*c1 - b1*c2
        den = a1*c2 - a2*c1
        # We'll assume the caller ensures slopes are not identical (den != 0).
        # We want den>0 for consistent fraction:
        if den < 0:
            den = -den
            num = -num
        return (num, den)  # rational intersection y = num/den

    # Compare two rationals (n1/d1) <= (n2/d2):
    def leq(r1, r2):
        n1, d1 = r1
        n2, d2 = r2
        # we assume d1>0, d2>0
        return n1*d2 <= n2*d1

    from math import floor
    
    outputs = []
    
    for _testcase in range(t):
        N = int(input_data[idx]); idx+=1
        lines = []
        for _ in range(N):
            A = int(input_data[idx])
            B = int(input_data[idx+1])
            C = int(input_data[idx+2])
            idx += 3
            # line => f(y) = (a*y + b)/c, with a = -B, b = (C-1), c = A
            # slope = a/c = -B / A (negative).
            lines.append( (-B, C-1, A) )
        
        # If N=0 (should not happen due to constraints), answer=0
        if N == 0:
            outputs.append('0')
            continue
        
        # Remove lines with duplicate slopes, keeping only the one with the smallest intercept.
        # If slope is the same => a1/c1 == a2/c2 => a1*c2 == a2*c1 => keep the line with smaller b/c.
        # i.e. keep the line that in real value b1/c1 <= b2/c2 for min envelope
        # Actually we want the line that yields the smaller f(y). For same slope, the difference is constant (b1/c1 - b2/c2).
        # If b1/c1 > b2/c2, then line1 is always above line2. We want to keep line2 for min.
        # We'll do a sort and then filter.
        
        # Sort by (slope ascending, then intercept ascending).
        # slope ascending means (a1/c1) < (a2/c2) => a1*c2 < a2*c1
        # if tie => compare b1/c1 vs b2/c2 => b1*c2 < b2*c1
        lines.sort(key=lambda x: (x[0]/x[2]))  # a quick sort by slope as float.  (Safe enough, though all are negative.)
        # But to be perfectly safe (avoid float issues), do an integer compare in a custom sort:
        # We'll implement a stable approach below if needed at scale, but Python's float should be okay in practice
        # for B/A up to 1e9 since double has 53 bits of precision.  We'll proceed for brevity.
        
        filtered = []
        # We'll do a single pass removing lines with same slope but higher intercept:
        i = 0
        while i < N:
            j = i+1
            (a_i, b_i, c_i) = lines[i]
            # We'll keep track of the best (a_i, b_i, c_i) among those that share the slope.
            # "best" = the one with minimum b/c in real sense => b1/c1 < b2/c2 => b1*c2 < b2*c1
            # For all lines with same slope, choose the smallest intercept.
            # slope_i = a_i/c_i
            best = (a_i, b_i, c_i)
            while j < N:
                (a_j, b_j, c_j) = lines[j]
                # check if slope_i == slope_j => a_i/c_i == a_j/c_j => a_i*c_j == a_j*c_i
                if a_i*c_j == a_j*c_i:
                    # same slope
                    # keep the one with smaller b/c
                    # compare b_i/c_i < b_j/c_j => b_i*c_j < b_j*c_i
                    if b_j*c_i < b_i*c_j:
                        best = (a_j, b_j, c_j)
                    j += 1
                else:
                    break
            filtered.append(best)
            i = j
        
        # Now build the "convex hull" of lines in ascending slope for min queries in ascending x.
        # The standard "LiChao / CHT" approach in incremental form:
        hull = []  # will store tuples (a,b,c, x_start) where x_start is rational intersection with previous
        import math
        
        def add_line(a, b, c):
            # new line
            new_line = (a, b, c)
            # intersection with top
            if not hull:
                # first line => x_intersect = -inf
                # store as (a,b,c, ('-inf',1))
                hull.append((a, b, c, (-10**30, 1)))  # "very negative"
            else:
                # pop while intersection <= top.x_start
                while True:
                    a_top, b_top, c_top, x_top = hull[-1]
                    # check if parallel => won't happen here, we already filtered slopes
                    # compute intersection with top
                    num = (b_top*c - b*c_top)
                    den = (a_top*c - a*c_top)
                    if den == 0:
                        # parallel in principle, but we shouldn't see it if well filtered
                        # skip or break
                        # in practice, just treat it as "no valid intersection"
                        # We'll break to not cause infinite loop
                        # But ideally we never get here due to filtering
                        # so let's remove the line if the new one is always better or vice versa
                        # For safety:
                        hull.pop()
                        if not hull:
                            # then push new_line as first
                            hull.append((a, b, c, (-10**30, 1)))
                            return
                        continue
                    if den < 0:
                        den = -den
                        num = -num
                    inter = (num, den)
                    # if inter <= x_top => pop
                    if leq(inter, x_top):
                        hull.pop()
                        if not hull:
                            # push as first
                            hull.append((a, b, c, (-10**30, 1)))
                            return
                    else:
                        # set new_line's x_start = inter
                        hull.append((a, b, c, inter))
                        return

        # Re-sort to ensure strictly ascending slope (a/c) after the "best" intercept filter.
        # We'll apply the same float-based sort or integer-based if needed.
        filtered.sort(key=lambda x: (x[0]/x[2]))
        
        hull.clear()
        for (a_i, b_i, c_i) in filtered:
            add_line(a_i, b_i, c_i)
        
        # Now we have a hull array of size up to len(filtered).
        # The line hull[i] = (a_i,b_i,c_i, x_i), is valid from x_i to x_{i+1}, 
        # where x_0 = (-inf,1), x_{len(hull)}= +inf in concept.
        
        # Next we find Y_max.  We only sum for y=1..Y_max where L(y) >= 1.
        # L(y) = min(lines) => for line i => (a_i*y + b_i)/c_i >= 1 => a_i*y + b_i >= c_i
        # => -B_i*y + (C_i-1) >= A_i => B_i*y <= (C_i-1)-A_i => y <= ((C_i-1)-A_i)/B_i for each i.
        # So Y_max = floor( min_i( ((C_i-1)-A_i)/B_i ) ).
        # If this <1 => answer=0
        # Implementation: each original constraint is (A,B,C), so we do:
        # y <= ((C-1) - A)/ B. We must do it for all constraints.  We have them in lines as (a=-B, b=C-1, c=A).
        # But let's just use the original input again or re-derive from the line:
        
        # Actually let's re-parse from filtered lines if we want.  Each line corresponds to some original constraint,
        # but we might have lost some lines if they were never the minimal.  However, that doesn't override the constraint
        # because losing them in the hull means there's a stricter line anyway.  So let's be safe and use the original constraints.
        
        min_ratio = None
        for (A_i, B_i, C_i_minus_1) in [(l[2],-l[0],l[1]+1) for l in filtered]:
            # We have reversed transformation: a=-B, b=C-1, c=A
            # so B_i = -a, C_i = b+1, A_i = c
            # y <= (C_i - 1 - A_i)/ B_i
            numerator = (C_i_minus_1 - A_i)
            denom = B_i  # B_i>0
            # If numerator < 0 => no positive y satisfies => region might be empty => Y_max < 0 => sum=0
            # We'll manage that by taking min_ratio = floor( something ) if positive
            import math
            if denom <= 0:
                # This should not happen because B>0 => but let's be safe
                # leads to no constraint or infinite. We'll skip
                continue
            val = numerator/denom
            if min_ratio is None or val < min_ratio:
                min_ratio = val
        
        if min_ratio is None:
            # means maybe no constraints or B=0 somewhere? (not in valid input though)
            # if there's no real constraint, we might have infinite? But problem states result is finite, so 0.
            outputs.append('0')
            continue
        
        Y_max_fl = int(min_ratio)  # floor
        # If min_ratio<1 => Y_max_fl <=0 => no positive y => answer=0
        if Y_max_fl < 1:
            outputs.append('0')
            continue
        
        # Now sum over y=1..Y_max_fl of floor( hull_min(y) ), 
        # where hull_min(y) = min_i f_i(y) = min over lines in hull. 
        # But in piecewise form, line i is valid from x_i to x_{i+1}.
        
        ans = 0
        # We'll define an array of intersection boundaries from the hull:
        # hull[i] = (a_i,b_i,c_i, x_start_i) => valid from x_start_i included up to next x_start (excluded).
        # But we actually do half-open intervals [ x_start_i, x_start_{i+1} ).
        
        # We'll iterate i=0.. len(hull)-1; define next boundary as x_start_{i+1} if i+1 < len(hull) else +∞
        
        def frac_floor(num, den):
            # floor(num/den) for den>0
            # handle negative num carefully
            return num // den if ((num >= 0) == (den > 0)) else -((-num)//den)
        
        import math
        
        for i in range(len(hull)):
            a_i, b_i, c_i, x_i = hull[i]
            if i+1 < len(hull):
                x_next = hull[i+1][3]  # the next line's x_start
            else:
                x_next = (10**30, 1)  # +inf
                
            # line i is valid for y in [ x_i, x_next ).
            # x_i is a fraction, x_next is a fraction
            start_y = frac_floor(x_i[0], x_i[1])  # floor of x_i
            # But the line is valid for y >= x_i, so the first integer y that is >= x_i is start_y+1 if x_i is integer, else start_y + ??? 
            # Actually if x_i is an integer fraction, line i is valid from that y onward. Typically in the standard CHT for min queries,
            # the transitions are "for x >= x_start_i, line i is better." So let's define the smallest integer y >= x_i:
            if x_i[0]>=0:
                # y_min = ceil(x_i)
                # ceil(a/b) = (a + b -1)//b if a>0
                y_min = (x_i[0] + x_i[1] -1)// x_i[1]
            else:
                # if x_i<0, then y_min can be 0 or 1 if we only consider positive
                y_min = 0  # means effectively 1 if we want positive domain
            
            end_y_f = x_next  # up to but not including x_next
            # last integer y that is < x_next => floor(x_next) - 1 if x_next is integer? Actually we use floor(x_next)-1 for open?
            # Actually if line i is valid in [ x_i, x_next ), we want y < x_next => y <= floor(x_next)-1 if x_next is integer, else y <= floor(x_next).
            # So let's define y_max = floor(x_next)-1 if x_next is integer, else floor(x_next).
            
            # We'll get floor_xn = floor(num/den):
            floor_xn = frac_floor(end_y_f[0], end_y_f[1])  # floor of x_next
            # The line i is valid for y < x_next => if x_next is an integer fraction (say 5 exactly), we do y<5 => y<=4 => floor_xn -1=4
            # So we check if end_y_f is exactly an integer:
            if end_y_f[0] % end_y_f[1] == 0:
                # x_next is integer
                y_max = floor_xn - 1
            else:
                y_max = floor_xn
            
            # Now the actual integer domain is y in [max(1,y_min).. min(Y_max_fl, y_max)].
            seg_start = max(1, y_min)
            seg_end = min(Y_max_fl, y_max)
            if seg_start <= seg_end:
                ans += sum_floor_line(a_i, b_i, c_i, seg_start, seg_end)
        
        outputs.append(str(ans))
    
    print('
'.join(outputs))