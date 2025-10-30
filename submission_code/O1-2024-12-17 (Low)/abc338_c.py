def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    Q = list(map(int, input_data[1:1+N]))
    A = list(map(int, input_data[1+N:1+2*N]))
    B = list(map(int, input_data[1+2*N:1+3*N]))

    # ------------------------------------------------------------------------
    # We want to maximize (x + y) subject to:
    #   For each i = 1..N:  A_i*x + B_i*y <= Q_i, and x >= 0, y >= 0.
    #
    # This is a 2-variable integer linear programming problem with N constraints.
    # N <= 10, which allows us to consider the "corners" of the feasible region.
    #
    # The feasible region is the intersection of half-planes in the x-y plane:
    #   x*A_i + y*B_i <= Q_i  (i=1..N),
    # plus x >= 0, y >= 0.
    #
    # In 2D, the maximum integer solution of x+y almost always lies at or near
    # an intersection of two of these constraint boundary lines (or with axes).
    #
    # We will:
    # 1. Collect all relevant intersection points:
    #    - Intersection with x=0 or y=0 for each constraint.
    #    - Intersections of each pair of constraints if not parallel.
    # 2. Around each intersection (which may be fractional), we will consider
    #    the "nearby" integer lattice points (floor/ceil in each coordinate).
    # 3. Keep all candidate points in a set, filter out those that do not satisfy
    #    all constraints (including non-negativity).
    # 4. Track the maximum x + y among all feasible points.
    #
    # This approach is O(N^2) in intersections, which is fine for N <= 10.
    # ------------------------------------------------------------------------

    # First build a list of constraints in the form (A_i, B_i, Q_i).
    constraints = []
    for i in range(N):
        constraints.append((A[i], B[i], Q[i]))

    # A helper function to check feasibility of (x, y):
    def feasible(x, y):
        if x < 0 or y < 0:
            return False
        for (a, b, q) in constraints:
            # a*x + b*y <= q must hold
            if a*x + b*y > q:
                return False
        return True

    # We'll collect candidate points in a set to avoid duplicates:
    candidates = set()

    # 1) Add (0, 0) as a candidate
    candidates.add((0, 0))

    # 2) For each constraint i, add intersections with x=0 or y=0, if valid.
    #    x=0 => B_i*y <= Q_i => if B_i>0 => y <= Q_i/B_i
    #    y=0 => A_i*x <= Q_i => if A_i>0 => x <= Q_i/A_i
    for (a, b, q) in constraints:
        if a == 0 and b == 0:
            # This constraint doesn't actually limit anything if q >= 0
            # If q<0 it is infeasible, but that shouldn't happen given constraints
            continue
        if a == 0:
            # Then we only have b*y <= q
            if b > 0:
                y0 = q // b
                candidates.add((0, y0))
        else:
            # a>0 or a<0, but per constraints 0<=A_i <=1e6, at least one A_i >=1
            # we only care if a>0 for x=0 boundary?
            if a > 0:
                x0 = q // a
                candidates.add((x0, 0))
        if b == 0:
            # Then we only have a*x <= q
            if a > 0:
                x0 = q // a
                candidates.add((x0, 0))
        else:
            # b>0
            y0 = q // b
            candidates.add((0, y0))

    # 3) For each pair of constraints, find the intersection of lines:
    #    a1*x + b1*y = q1
    #    a2*x + b2*y = q2
    #    We'll get possibly fractional (x,y). We'll take floor/ceil combos.
    import math

    for i in range(N):
        a1, b1, q1 = constraints[i]
        for j in range(i+1, N):
            a2, b2, q2 = constraints[j]
            # Solve:
            #   a1*x + b1*y = q1
            #   a2*x + b2*y = q2
            # in real numbers if det != 0
            det = a1*b2 - a2*b1
            if det == 0:
                # parallel or coincident => skip
                continue
            # x = (q1*b2 - q2*b1)/det
            # y = (a1*q2 - a2*q1)/det
            # We'll keep them as floats or do integer-based rational handling
            # Then consider floor/ceil around those points.
            x_float = (q1*b2 - q2*b1) / det
            y_float = (a1*q2 - a2*q1) / det

            # Let's define a small function to gather all integer combos near x_float, y_float
            # We'll only check floor and ceil of each.
            x_candidates = [math.floor(x_float), math.ceil(x_float)]
            y_candidates = [math.floor(y_float), math.ceil(y_float)]
            for xx in x_candidates:
                for yy in y_candidates:
                    candidates.add((xx, yy))

    # Now we have a set of candidate integer points. Let's filter them by feasibility
    # and keep track of the maximum x+y.
    ans = 0
    for (x, y) in candidates:
        if x < 0 or y < 0:
            continue
        if feasible(x, y):
            ans = max(ans, x+y)

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()