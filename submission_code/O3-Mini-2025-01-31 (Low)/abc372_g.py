def main():
    import sys,math
    sys.setrecursionlimit(3000000)
    data = sys.stdin.buffer.read().split()
    t = int(data[0])
    index = 1

    # Floor Sum: Compute sum_{0<=i<n} floor((a*i+b)/m)
    # This implementation is taken from a known algorithm.
    def floor_sum(n, m, a, b):
        # returns sum_{0<=i<n} floor((a*i+b)/m)
        # m > 0 assumed.
        total = 0
        while True:
            if a >= m or b >= m:
                total += (n - 1) * n * (a // m) // 2
                total += n * (b // m)
                a %= m
                b %= m
            y = a * n + b
            if y < m:
                return total
            n, m, a, b = y // m, a, m, y % m

    # The envelope will be a list of segments.
    # Each segment corresponds to an interval [L, R] (real numbers) where the envelope is given by
    #   f(x) = (-A/B)*x + ((C-1)/B)
    # We represent the active line by the tuple (A, B, C) (original parameters, note A,B,C >0)
    # and the segment endpoint x_left.
    # We maintain that the envelope is the lower envelope (i.e. the pointwise minimum) over x >= 1.
    #
    # The algorithm: sort lines by slope, i.e., by m = -A/B. Because A,B>0, m is negative.
    # In order to get the envelope we sort by decreasing value of m (i.e. by A/B increasing).
    # Then, using standard "Convex Hull Trick" technique, we compute the envelope segments.
    def envelope(lines):
        # lines: list of (A, B, C) corresponding to line f(x)= (C-1-A*x)/B.
        # For convenience, we convert each to (m, c, A, B, C)
        new_lines = []
        for (A, B, C) in lines:
            # slope m = -A/B and intercept c = (C-1)/B; we store as a tuple.
            new_lines.append(( -A/B, (C-1)/B, A, B, C ))
        # sort by slope, then by intercept:
        new_lines.sort(key=lambda tup: (tup[0], tup[1]))
        # Now build the lower envelope
        env = []  # each element: (x_start, line) where line = (m, c, A, B, C)
        # x_start is the x value where this line becomes active.
        # For domain x starting at 1.
        def inter(line1, line2):
            # compute x such that line1(x) == line2(x)
            m1, c1, _, _, _ = line1
            m2, c2, _, _, _ = line2
            # m1*x+c1 = m2*x+c2 => x = (c2-c1)/(m1-m2)
            return (c2 - c1) / (m1 - m2)
        for line in new_lines:
            # remove lines that never contribute:
            if env and abs(env[-1][1][0] - line[0]) < 1e-12:
                # same slope: keep the one with lower intercept
                if line[1] > env[-1][1][1]:
                    continue
                else:
                    env.pop()
            while env:
                x_int = inter(env[-1][1], line)
                if x_int <= env[-1][0]:
                    env.pop()
                else:
                    break
            if not env:
                x_start = 1.0
            else:
                x_start = inter(env[-1][1], line)
                if x_start < 1.0:
                    x_start = 1.0
            env.append((x_start, line))
        # Now, convert env into a list of segments: each with integer endpoints [L, R] and line parameters.
        segments = []
        for i in range(len(env)):
            L = env[i][0]
            if i+1 < len(env):
                R = env[i+1][0]
            else:
                R = float('inf')
            # We want integer x in [ceil(L), floor(R)] eventually.
            segments.append((L, R, env[i][1]))
        return segments

    # For one test case, we use envelope technique to sum over x from 1 to X_max the value of:
    # f(x) = min_i floor((C_i-1-A_i*x)/B_i)
    # Our domain is x positive and we require f(x)>=1.
    # We compute X_max from the envelope: The largest x such that f(x) >= 1.
    # Notice that for a segment with active line (A, B, C), we have
    #   f(x) = floor((C-1-A*x)/B)
    # so f(x) >= 1 means (C-1-A*x) >= B, i.e. x <= (C-1 - B)//A  (using floor division).
    # Thus, for each constraint, the condition for x is x <= (C - B - 1)//A.
    def solve_case(n, triples):
        # Compute an overall trivial x_max from each constraint: 
        Xmax = 10**18
        for (A, B, C) in triples:
            Xmax = min(Xmax, (C - B - 1) // A)
        if Xmax < 1:
            return 0
        # Build envelope over lines from all constraints.
        segs = envelope(triples)
        total = 0
        curr = 1
        # Process each segment (which is valid on a real interval). In each segment,
        # the formula is f(x)= floor((C-1-A*x)/B) with the active line's parameters.
        for (L, R, line) in segs:
            m, c, A, B, C = line
            # The segment applies for real x in [L, R). We want integer x in [max(curr, ceil(L)), min(Xmax, floor(R - 1e-12))]
            start = max(curr, int(math.ceil(L)))
            if R == float('inf'):
                end = Xmax
            else:
                end = min(Xmax, int(math.floor(R - 1e-12)))
            if start > end:
                continue
            # In this interval, for an integer x, f(x)= floor((C-1-A*x)/B).
            # Let’s denote for simplicity: for integer x,
            #    f(x) = (C-1-A*x) // B.
            # We want to sum S = sum_{x = start}^{end} ((C-1-A*x) // B).
            #
            # We now compute S using a O(log(...)) algorithm using the standard floor sum algorithm.
            # We wish to compute S = sum_{x = start}^{end} g(x) where g(x)= (C-1-A*x) // B.
            # Set n = end - start + 1.
            # Write for x in [start, end]:
            #   g(x) = ( (C-1) - A*x) // B.
            # Let X = x - start, so x = start+X with X from 0 to n-1.
            # Then:
            # g(x) = ( (C-1) - A*start - A*X) // B.
            # Let P = (C-1 - A*start). Then g(x) = (P - A*X)//B.
            n_count = end - start + 1
            P = C - 1 - A * start
            # We want sum_{X=0}^{n_count-1} floor((P - A*X)/B).
            # Note that A, B, P are nonnegative? (They may be zero? Actually A>=1 and B>=1 and P can be negative.)
            # But note: by our domain choice f(x) >=1 so (C-1-A*x) >= B so P >= B.
            # However, A is positive so the summand is decreasing.
            # We can’t directly use the standard floor_sum because here A appears with a negative sign.
            #
            # We use the following idea: For each X, floor((P - A*X)/B) = k.
            # We can compute the sum by iterating over X in O(√n) steps. However n_count might be large.
            #
            # Instead we use the standard “floor sum” algorithm with a twist.
            # Observe that:
            #    floor((P - A*X)/B) = P//B - ceil((A*X + r)/B)
            # where r = (P % B) and using the identity
            #    floor((P - A*X)/B) = P//B - ((A*X + (B - 1 - r)) // B) - 1  (with proper adjustments)
            # Instead, here we apply a standard iterative method.
            #
            # We can sum by binary–jumping in X. (This is similar to summing the floor of a linear function.)
            def sum_floor(n, A, B, P):
                # Compute sum_{X=0}^{n-1} floor((P - A*X)/B)
                res = 0
                x = 0
                while x < n:
                    # Current value:
                    cur = (P - A*x) // B
                    # if cur is negative, then break, but that shouldn’t happen here.
                    # Find maximum y such that for all X in [x, y] the value remains = cur.
                    # Solve: (P - A*y) // B = cur.
                    # That is, A*y <= P - cur*B, so y_max = floor((P - cur*B)/A)
                    if A == 0:
                        y = n - 1
                    else:
                        y = (P - cur * B) // A
                    y = min(n - 1, y)
                    cnt = y - x + 1
                    res += cur * cnt
                    x = y + 1
                return res
            S_seg = sum_floor(n_count, A, B, P)
            total += S_seg
            curr = end + 1
            if curr > Xmax:
                break
        return total

    out_lines = []
    for _ in range(t):
        n = int(data[index]); index += 1
        triples = []
        for i in range(n):
            A = int(data[index]); B = int(data[index+1]); C = int(data[index+2])
            index += 3
            triples.append((A,B,C))
        out_lines.append(str(solve_case(n, triples)))
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()