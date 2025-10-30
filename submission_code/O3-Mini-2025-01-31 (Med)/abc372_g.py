def main():
    import sys, math
    data = sys.stdin.read().strip().split()
    if not data:
        return
    t = int(data[0])
    idx = 1
    out_lines = []
    # floor_sum(n, m, a, b) returns sum_{j=0}^{n-1} floor((a*j + b) / m)
    # It assumes n, m, a, b are nonnegative integers and m > 0.
    def floor_sum(n, m, a, b):
        s = 0
        while True:
            if a >= m:
                s += (n - 1) * n * (a // m) // 2
                a %= m
            if b >= m:
                s += n * (b // m)
                b %= m
            y = a * n + b
            if y < m:
                return s
            n, m, a, b = y // m, m, y % m, a

    # Explanation of our approach:
    # For each inequality A*x + B*y < C, writing it as y <= floor((C-1 - A*x)/B)
    # the feasible (x,y) must satisfy for all constraints:
    #    y <= min_i floor((C_i - 1 - A_i*x)/B_i)
    #
    # Also, for a feasible y≥1 we must have C_i - 1 - A_i*x >= B_i for every i,
    # so x is bounded above by:
    #    x_i_max = floor((C_i - 1 - B_i) / A_i)
    # and overall x_max = min_i x_i_max.
    #
    # Next, for each constraint we put the line:
    #    L_i(x) = (C_i - 1)/B_i - (A_i/B_i)*x.
    # The allowed y for a given x is the floor of the minimum of these lines.
    # We then compute the lower envelope (pointwise minimum) of these affine functions.
    
    eps = 1e-12
    for _ in range(t):
        if idx >= len(data):
            break
        n = int(data[idx]); idx += 1
        constraints = []
        global_xmax = None
        feasible = True
        for i in range(n):
            A = int(data[idx]); B = int(data[idx+1]); C = int(data[idx+2])
            idx += 3
            T = C - 1  # we set T = C-1 so that A*x + B*y <= T is equivalent to A*x+B*y < C (for integers)
            # For feasibility (y>=1 exists) we need T - A*x >= B 
            if T < B:
                feasible = False
            else:
                # maximum x for which constraint i permits some y>=1:
                xi_max = (T - B) // A
                if global_xmax is None or xi_max < global_xmax:
                    global_xmax = xi_max
            constraints.append((A, B, C))
        if (not feasible) or (global_xmax is None) or (global_xmax < 1):
            out_lines.append("0")
            continue
        Xmax = global_xmax  # domain for x is 1 <= x <= Xmax

        # Build list of lines for the lower envelope.
        # For each constraint (A,B,C) we define:
        #   m = A/B   and c = (C-1)/B
        # so that L(x) = c - m*x.
        lines = []
        for (A, B, C) in constraints:
            T = C - 1
            m_val = A / B
            c_val = T / B
            lines.append((m_val, c_val, A, B, T))
        # Sort lines by slope, breaking ties by c.
        lines.sort(key=lambda x: (x[0], x[1]))
        
        # Build the lower envelope. The idea is that each line, when added in order of increasing slope,
        # becomes active from some x value (its "breakpoint"). We maintain a hull of segments.
        # Each hull element will be a tuple:
        #    (start, m, c, A, B, T)
        # meaning that its line is optimal for x >= start (up to the next breakpoint).
        hull = []
        for (m_val, c_val, A, B, T) in lines:
            # if same slope as last, take the one with lower c (i.e. lower function values)
            if hull and abs(m_val - hull[-1][1]) < eps:
                if c_val < hull[-1][2] - eps:
                    hull.pop()
                else:
                    continue
            if not hull:
                start_x = -math.inf
            else:
                # Intersection with last line: For line L_last: c_last - m_last*x, and current line: c_val - m_val*x,
                # we solve: c_last - m_last*x = c_val - m_val*x  =>  x = (c_last - c_val) / (m_last - m_val)
                last = hull[-1]
                inter = (last[2] - c_val) / (last[1] - m_val)
                start_x = inter
                # If this new intersection is not to the right of the last segment's start,
                # then pop the last segment.
                while hull and start_x <= hull[-1][0] + eps:
                    hull.pop()
                    if hull:
                        last = hull[-1]
                        start_x = (last[2] - c_val) / (last[1] - m_val)
            hull.append((start_x, m_val, c_val, A, B, T))
        
        # Now, the envelope is given by hull.
        # For x in [1, Xmax] we want to add, for each integer x,
        #    f(x) = floor((T - A*x)/B) 
        # where (A,B,T) are taken from the hull element active at x.
        # The hull segments: hull[i] is active for x in [hull[i][0], hull[i+1][0])
        # (with the last one active for x >= hull[last][0]).
        # Because x is an integer and our domain is [1, Xmax], we “clip” the segments.
        # One trick is to sum over an interval [L,R] quickly.
        #
        # For a fixed line (A,B,T) valid in an interval x ∈ [L,R], note that
        #    S = Σₓ₌ₗ^(R) floor((T - A*x)/B)
        # Changing variable: let x = L + j, j=0,...,n-1 with n = R-L+1.
        # Then S = Σ_{j=0}^{n-1} floor((T - A*(L+j))/B)
        #   = Σ_{j=0}^{n-1} floor((T - A*R + A*(n-1 - j))/B)
        # replacing j by i = n-1 - j we get:
        #    S = Σ_{i=0}^{n-1} floor((A*i + (T - A*R)) / B)
        # and the standard floor_sum(n, B, A, b) computes this sum.
        #
        # Now we iterate over the segments in our hull
        total = 0
        # helper: sum over integers x in [Lx, Rx] for a line with parameters (A, B, T)
        def sum_segment(Lx, Rx, A, B, T):
            n = Rx - Lx + 1
            b = T - A * Rx  # b = T - A*R
            return floor_sum(n, B, A, b)
        
        # For each hull segment determine the integer x-interval (within [1,Xmax]) when this line is active.
        for i in range(len(hull)):
            seg_start_val = hull[i][0]
            if seg_start_val == -math.inf:
                L_int = 1
            else:
                # if seg_start_val is nearly an integer, then we use its ceiling.
                L_int = math.ceil(seg_start_val - eps)
                if L_int < 1:
                    L_int = 1
            if i + 1 < len(hull):
                next_break = hull[i+1][0]
                # The current line is active for x < next_break.
                # So the largest integer x that is strictly less than next_break is:
                r_floor = math.floor(next_break + eps)
                # if next_break is nearly an integer, then that integer belongs to the next segment.
                if abs(next_break - r_floor) < eps:
                    R_int = r_floor - 1
                else:
                    R_int = r_floor
            else:
                R_int = Xmax
            if R_int > Xmax:
                R_int = Xmax
            if L_int > R_int:
                continue
            total += sum_segment(L_int, R_int, hull[i][3], hull[i][4], hull[i][5])
        out_lines.append(str(total))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()