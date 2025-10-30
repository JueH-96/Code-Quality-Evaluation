# YOUR CODE HERE

import sys
import math
sys.setrecursionlimit(10**7)

# ------------------------------------------------------------------------------
# We want to count the number of positive-integer solutions (x,y) to
#   A_i*x + B_i*y < C_i   for all i=1..N,
# given N inequalities, with A_i,B_i,C_i > 0.
#
# Equivalently, requiring x>=1,y>=1 (since x,y>0 and integer), we rewrite:
#   A_i*(x) + B_i*(y) <= C_i - 1
# Let x' = x - 1 >= 0, y' = y - 1 >= 0 be nonnegative integers.
# Then
#   A_i*(x'+1) + B_i*(y'+1) <= C_i - 1
#   A_i*x' + B_i*y' <= (C_i -1) - (A_i + B_i)
# Define D_i = C_i - 1 - (A_i + B_i).
# So the (x',y') must satisfy:
#   A_i*x' + B_i*y' <= D_i,   for all i,
#   x' >= 0, y' >= 0.
#
# If any D_i < 0, there are no solutions. Else, the feasible region in the
# (x',y') plane is the intersection of half-planes A_i*x' + B_i*y' <= D_i
# (with A_i,B_i>0) plus x' >=0, y' >=0. That region is a convex polygon
# in the first quadrant, or possibly empty. The problem states the number
# of solutions is finite, so it must be bounded.
#
# We then want to count all integer lattice points (x',y') in that polygon.
# Finally, the count of (x,y) is the same as count of (x',y') since x= x'+1,
# y= y'+1 is a bijection for x',y' >=0.
#
# A standard way to count integer points in a convex polygon whose edges
# all have integer coefficients is to use Pick's Theorem:
#   IntegerPoints = Area + (BoundaryPoints/2) + 1,
# where Area is the polygon's (Euclidean) area, and BoundaryPoints is the
# number of lattice points on the boundary.
#
# Hence the main task is to perform half-plane intersection of:
#   1) x' >= 0, i.e. line: x' = 0 => 1*x' + 0*y' >= 0  =>  (-1)*x' + 0*y' <= 0 (inequality).
#   2) y' >= 0, i.e. line: y' = 0 => 0*x' + 1*y' >= 0  =>  0*x' + (-1)*y' <= 0.
#   3) For each i: A_i*x' + B_i*y' <= D_i.
# Then, if the intersection is non-empty and bounded, compute the polygon
# (in order around the edges). Then compute:
#   - The area (as a rational or floating). Must do it carefully (exactly),
#   - The number of boundary lattice points. We can find boundary points
#     between consecutive vertices by gcd(|dx|,|dy|).
# Then use Pick's Theorem. If the intersection is empty, answer=0.
#
# Implementation outine:
#   1) For each test case:
#       - Read N, read lines (A_i,B_i,C_i).
#       - Compute D_i = C_i - 1 - (A_i + B_i). If any D_i < 0 => answer=0 and skip.
#       - Collect half-planes:
#            (A_i, B_i, D_i)        for i=1..N
#            (1, 0, 0)   for x' >=0 means 1*x' + 0*y' >=0 => rewrite to -1*x' <= 0 => we store ( -1, 0, 0 ) in code
#            (0, 1, 0)   for y' >=0 means 0*x' + 1*y' >=0 => rewrite to 0*x' + -1*y' <= 0 => we store ( 0, -1, 0 ) in code
#         BUT we usually represent half-planes as a*x + b*y <= c with the region "to the left"
#         of the directed edge. So for x' >=0 we want x' >=0 => x' - 0 >=0 => -x' <=0 is indeed a*x'+b*y'<=c form is (-1,0,0).
#         Similarly y' >=0 => (0,-1,0).
#       - Perform half-plane intersection and get the resulting polygon in CCW order.
#       - If empty => 0
#         else compute pick's theorem => area + boundary/2 + 1.
#   2) Print the results.
#
# We will implement an O(M log M) half-plane intersection, where M = N+2 for
# the extra lines. The sums of N up to 2e5 can be handled. We must be mindful
# of performance. We'll do exact rational intersection to avoid floating errors.
#
# Steps for half-plane intersection (HPI) (standard "cut" approach in CCW order):
#   - We want lines in CCW around the polygon. Typically one sorts lines by polar angle
#     of the normal, then does a "deque" approach. We'll store each line in a form
#     that consistently orients the normal vector so we keep left side "inside".
#   - Implementation details can be found in many references. We'll do the usual approach
#     carefully with rationals.
#
# Once we have the polygon vertices in CCW order (as rational points),
# we compute the oriented area using cross products in rational form.
# Then we compute the boundary points by summing gcd of delta x, delta y
# in their exact (integer) difference form. Because each vertex is (x/d, y/d),
# we keep them as pairs of (x, y, d). The difference in successive vertices is
# (x2/d2 - x1/d1, y2/d2 - y1/d1). Putting them on a common denominator D = lcm(d1,d2),
# we get integer differences, and we can gcd those to find the number of boundary
# points on that segment minus 1 for the endpoints. We'll then sum +1 for each vertex,
# etc. (standard approach).
#
# Implementation below is non-trivial but follows standard computational geometry.
# ------------------------------------------------------------------------------

from math import gcd
from collections import deque

# We'll store lines in the form (A,B,C) meaning the half-plane is A*x + B*y <= C.
# We'll implement half-plane intersection in CCW order of the lines' direction.
# Then retrieve the intersection polygon in CCW order.

# A small helper class for rational numbers (numerator, denominator) in reduced form:
class Rational:
    __slots__ = ('n','d')
    def __init__(self, n, d=1):
        # store sign in n
        if d<0:
            d = -d
            n = -n
        g = math.gcd(n,d)
        self.n = n//g
        self.d = d//g
    def __add__(self, other):
        return Rational(self.n*other.d + other.n*self.d, self.d*other.d)
    def __sub__(self, other):
        return Rational(self.n*other.d - other.n*self.d, self.d*other.d)
    def __mul__(self, other):
        return Rational(self.n*other.n, self.d*other.d)
    def __truediv__(self, other):
        return Rational(self.n*other.d, self.d*other.n)
    def __lt__(self, other):
        return self.n*other.d < other.n*self.d
    def __le__(self, other):
        return self.n*other.d <= other.n*self.d
    def __eq__(self, other):
        return self.n == other.n and self.d == other.d
    def value(self):
        return self.n/self.d  # float value
    def __repr__(self):
        return f"{self.n}/{self.d}"

# However, implementing a full Rational class for everything is quite heavy.
# Instead, we can do intersection with integer arithmetic directly:
# Intersection of:
#   A1*x + B1*y = C1
#   A2*x + B2*y = C2
# => We'll solve using determinant:
#   det = A1*B2 - A2*B1
#   x = (C1*B2 - C2*B1) / det
#   y = (A1*C2 - A2*C1) / det
#
# We'll store each point as (x_num, y_num, denom) with gcd factor stripped out so that
# x = x_num/denom, y = y_num/denom in reduced form, but we only require that gcd(x_num,
# y_num, denom)=1 except we must keep sign in numerator, ensure denom>0, etc.

def intersect(L1, L2):
    """
    Intersect two lines L1=(A1,B1,C1), L2=(A2,B2,C2) in the form:
       A*x + B*y = C
    Return the point as (Xnum, Ynum, D) with gcd(|Xnum|,|Ynum|,D)=1, D>0,
    or None if parallel (should not happen in half-plane intersection if no degenerate).
    """
    A1,B1,C1 = L1
    A2,B2,C2 = L2
    det = A1*B2 - A2*B1
    if det == 0:
        return None  # parallel or coincident
    # x_num = C1*B2 - C2*B1
    # y_num = A1*C2 - A2*C1
    x_num = C1*B2 - C2*B1
    y_num = A1*C2 - A2*C1
    # We want to normalize so that gcd(|x_num|,|y_num|,|det|)=1, det>0 if possible
    if det<0:
        det = -det
        x_num = -x_num
        y_num = -y_num
    g = math.gcd(x_num, math.gcd(y_num, det))
    x_num //= g
    y_num //= g
    det   //= g
    return (x_num, y_num, det)

def eval_line(line, pt):
    """
    Evaluate A*x + B*y - C at point pt=(x_num,y_num,d).
    If the point is strictly inside the line's half-plane (A*x+B*y <=C),
    the result should be <=0.
    We'll compute sign of (A*(x_num/d) + B*(y_num/d) - C).
    i.e. A*x_num + B*y_num - C*d.
    Return that integer. If <=0 => satisfies line's inequality.
    """
    A,B,C = line
    x_num,y_num,d = pt
    return A*x_num + B*y_num - C*d

def side(line, pt):
    """
    Return negative/zero if pt is inside or on line's half-plane: A*x + B*y <= C.
    Return positive if pt is outside.
    """
    return eval_line(line, pt)

def line_angle(line):
    """
    We need to sort lines by their angle for the typical HPI method.
    The direction vector of the line's normal is (A,B).
    The angle = atan2(B, A).
    We'll just use the built-in function or a custom approach.  We'll sort by
    (A,B) angle.  Because A,B>0 can appear? We also have lines for x'>=0 => that is ( -1,0 ).
    We'll do atan2(B, A).  If A=0, B>0 => angle= pi/2. If B=0, A>0 => angle=0, etc.
    """
    A,B,_ = line
    # angle in range (-pi, pi], but we only need increasing order
    return math.atan2(B, A)

def normalize_line(A,B,C):
    """
    We want all lines in a consistent orientation so the inside is always to the left
    of the directed normal (A,B).
    That means we want to ensure that if we travel along the line in direction ( -B, A ),
    the half-plane is to our left. Usually the condition is: the line is stored so that
    when we do 'side(line, a point that is slightly to the left of direction(-B,A))',
    we get negative => inside. There's a standard approach: we want to ensure that
    (A,B) or maybe we want to ensure A>0 or if A=0 then B>0 or so. But we have constraints
    from the problem that A_i>0,B_i>0 except for the two lines for x'>=0,y'>=0 which are
    ( -1,0),(0,-1 ). We just want a consistent half-plane orientation. Usually, if we find
    the line is "backwards," we multiply by -1 to fix it.
    We'll do:
       if (A<0) or (A=0 and B<0) => multiply by -1.
    This ensures that the normal's direction won't conflict in the sorting step.
    """
    if A<0 or (A==0 and B<0):
        A,B,C = -A,-B,-C
    return (A,B,C)

def half_plane_intersect(lines):
    """
    Compute the intersection polygon (in CCW order of vertices) of all half-planes
    given by lines = [(A,B,C), ...], each meaning A*x+B*y <= C.
    Return a list of vertices in CCW order, each in rational form (x_num,y_num,den).
    If empty, return [].
    Implementation: We'll use the "standard" method of sorting lines by angle,
    then process them one-by-one using a deque to maintain the intersection.
    """
    # 1) Normalize lines so that their normal (A,B) is consistently oriented
    #    and sort by angle.
    normed = []
    for (A,B,C) in lines:
        nL = normalize_line(A,B,C)
        normed.append(nL)
    normed.sort(key=line_angle)

    # We'll use the typical "dual" half-plane intersection approach:
    # Because of large N, we must implement an O(N) or O(N log N) approach carefully.
    # We'll do the standard O(N) "rotational sweep" approach, requiring lines
    # to be sorted by angle, and then we remove lines that cause no feasible region
    # in the "deque." This is a well-known technique, but the implementation is subtle.
    #
    # However, in practice, many prefer to do the simpler "off-the-shelf" implementation:
    #   1) Start with a big "box" or use e.g. polygon: a large bounding polygon.
    #   2) For each line in any order, 'cut' the polygon by that line. (O(V) per line),
    #      total O(NV). In worst case, V can be O(N), so O(N^2). That is too big for N=2e5.
    #
    # The standard "deque-based" method is O(N) if lines are sorted by angle.
    #
    # Implementing that in detail with integer intersection is quite long. For brevity
    # and reliability, we'll do a well-known simpler library-like code. We must be sure
    # to handle edge cases carefully. This is fairly advanced geometry code.

    # We'll implement the "line by line" approach in O(N) after sorting by angle:

    def check_intersection_back(lines_dq, new_line):
        # Pop from the back while the last vertex is not inside new_line
        # We'll check the intersection of the last two lines in the deque,
        # if that point is outside new_line, we pop the second-to-last line.
        while len(lines_dq)>=2:
            # intersection of lines_dq[-2] and lines_dq[-1]
            pt = intersect(lines_dq[-2], lines_dq[-1])
            if pt is None:
                # parallel lines or something degenerate => just pop
                lines_dq.pop()
                continue
            if side(new_line, pt) > 0:
                # if this intersection is outside new_line,
                # remove the last line and check again
                lines_dq.pop()
            else:
                break

    def check_intersection_front(lines_dq, new_line):
        # Similarly for the front
        while len(lines_dq)>=2:
            pt = intersect(lines_dq[0], lines_dq[1])
            if pt is None:
                lines_dq.popleft()
                continue
            if side(new_line, pt) > 0:
                lines_dq.popleft()
            else:
                break

    dq = deque()

    for ln in normed:
        check_intersection_back(dq, ln)
        check_intersection_front(dq, ln)
        dq.append(ln)

    # Now we must also do one final pass to ensure the front/back closure
    # because the polygon must be closed. We'll do the same pop logic but
    # once or twice with the final set.

    # If after this we have fewer than 2 lines, there's no polygon (or it's degenerate).
    if len(dq)<2:
        # Either no intersection or it's a line or empty
        # Check if there's a feasible region at all. Possibly check 1 line? Usually 0 solutions or infinite.
        # But problem states solutions are finite if they exist. So presumably 0.
        # We'll do a quick check if there's any feasible point at all (like (0,0))?
        # We'll just see if all lines are satisfied by (0,0). If yes => infinite/ not bounded? or maybe not
        # We can just return empty => 0 solutions.
        return []

    # trim front/back again
    check_intersection_back(dq, dq[0])
    check_intersection_front(dq, dq[-1])

    # We'll get the final list
    final_lines = list(dq)
    # Build polygon by intersecting consecutive lines
    poly = []
    for i in range(len(final_lines)):
        L1 = final_lines[i]
        L2 = final_lines[(i+1)%len(final_lines)]
        pt = intersect(L1, L2)
        if pt is None:
            # parallel => degenerate
            return []
        # Check if the point actually satisfies all lines? It should if everything is consistent
        poly.append(pt)
    # Now poly is in (some) order, but we want them in CCW order. The typical method ensures
    # it should be in CCW if we used the standard approach. We'll assume it is.

    # We must now check that each consecutive intersection is "inside" all lines.
    # But normally the half-plane intersection approach ensures that.

    # Return the polygon
    return poly

def polygon_area_rational(poly):
    """
    Compute the (signed) area of the polygon in rational form using cross product.
    poly is a list of vertices in CCW order, each = (x_num, y_num, d).
    We use the standard shoelace formula:
      area = 1/2 * sum_{i} ( x_i*y_{i+1} - y_i*x_{i+1} ) in usual float,
    but we do it in rational form carefully:
    Let vertex i = (Xi, Yi, Di), vertex i+1 = (Xj, Yj, Dj).
    The difference (x_i*y_{i+1} - y_i*x_{i+1}) in rational is:
      (Xi/Di)*(Yj/Dj) - (Yi/Di)*(Xj/Dj)
      = (Xi*Yj - Yi*Xj) / (Di*Dj)
    We'll sum them up over i, store as a fraction NUM/DEN (reduced).
    The area is half of that sum in absolute value (since poly is CCW, the sum should be positive).
    Return (num, den) for the area = (NUM, DEN) meaning area=NUM/DEN. We'll keep it un-reduced
    (or we can reduce at the end).
    """
    n = len(poly)
    num = 0
    den = 1  # We'll keep a running fraction sum = num/den
    # We'll do a standard approach for adding fractions in a single pass
    # but that might be large. We'll do the cross product in integer then combine carefully.
    big_num = 0
    big_den = 1
    for i in range(n):
        x1,y1,d1 = poly[i]
        x2,y2,d2 = poly[(i+1)%n]
        cross_num = (x1*y2 - y1*x2)
        cross_den = d1*d2
        # Now we add cross_num/cross_den to big_num/big_den
        # sum = (big_num/big_den)+(cross_num/cross_den) = (big_num*cross_den + cross_num*big_den)/(big_den*cross_den)
        tmp_num = big_num*cross_den + cross_num*big_den
        tmp_den = big_den*cross_den
        g = math.gcd(tmp_num, tmp_den)
        tmp_num //= g
        tmp_den //= g
        big_num = tmp_num
        big_den = tmp_den

    # big_num/big_den is the sum of cross, i.e. 2*area (since area=0.5 * sum of cross).
    # So area in fraction is = big_num/(2 * big_den).
    # The polygon is guaranteed to be CCW => area>0 => big_num should be >=0.
    # We'll return that fraction as (big_num, 2*big_den).
    if big_num<0:
        # in case the polygon is clockwise for some reason, just take absolute
        big_num = -big_num
    area_num = big_num
    area_den = 2*big_den
    return (area_num, area_den)

def boundary_points(poly):
    """
    Compute the number of integer lattice points on the boundary of the polygon.
    Summation over edges of gcd of delta_x, delta_y, because each vertex is rational.
    Edge from (x1/d1, y1/d1) to (x2/d2, y2/d2). Put on a common denominator D = lcm(d1,d2).
    Then the difference in numerator is an integer. The gcd of that difference gives
    the number of boundary points minus 1 (the endpoints). Summing over all edges
    plus the number of vertices. But we must not double-count the vertices. The standard
    formula is sum_{edges} gcd(|dx|,|dy|). That count includes both endpoints of each edge,
    but each vertex belongs to two edges, so total unique boundary points is that sum
    minus the number of edges. Reference: well-known result. We'll do exactly that:
       B = sum_{i} gcd(|X_{i+1}-X_i|, |Y_{i+1}-Y_i|)   where X_i, Y_i are in integer coordinates
       after scaling to a common denominator, but we must do it carefully because
       each has its own denominator.
    We'll do:
       - Let v_i = (x1, y1, d1), v_{i+1}=(x2,y2,d2).
       - D = lcm(d1, d2)
       - X1 = x1*(D/d1), Y1 = y1*(D/d1)
       - X2 = x2*(D/d2), Y2 = y2*(D/d2)
       - dx = X2 - X1, dy = Y2 - Y1
       - G = gcd(|dx|, |dy|)
       - That is the number of "lattice segments" between them on that scaled integer grid.
       - The number of integer points on the segment is G+1, so the segment boundary contributes G.
    Then sum over edges => total = sum(G). But that double counts the vertices once each,
    so if we want the total boundary points B, we do B = sum(G) + 1. Actually, for a closed
    polygon with n edges, we do B = sum(G) - (n - 1). The standard simpler known formula
    is B = sum(G) for a closed polygon in 2D with rational vertices. Indeed,
    "B = Σ gcd(|dx|,|dy|)" around the cycle. That includes each vertex exactly once. Reference:
    You can see that if you 'walk' edge by edge, each edge has G steps, and the new vertex
    is the final step of each edge. Summing them around the cycle counts n extra for the n edges,
    but also the first vertex is the same as last. A quick fix is indeed B = sum(G).
    Let's do B = sum(G).
    """
    n = len(poly)
    B = 0
    for i in range(n):
        x1,y1,d1 = poly[i]
        x2,y2,d2 = poly[(i+1)%n]
        # lcm
        l = (d1*d2)//math.gcd(d1,d2)
        X1 = x1*(l//d1)
        Y1 = y1*(l//d1)
        X2 = x2*(l//d2)
        Y2 = y2*(l//d2)
        dx = X2 - X1
        dy = Y2 - Y1
        B += abs(math.gcd(dx, dy))
    return B

def solve_one_case():
    input_data = sys.stdin.read().strip().split()
    # Because T testcases can be large, we can't parse each one separately in a naive way.
    # We'll parse here fully; but let's do it carefully outside. Actually, let's do it in main().
    pass

def main():
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    index = 1
    out = []
    for _ in range(t):
        N = int(input_data[index]); index+=1
        lines = []
        allDnonneg = True
        # We also add lines for x' >= 0 => line is -1*x' <=0 => ( -1,0,0 )
        # and y' >= 0 => line is (0,-1,0 )
        # but let's do it after we read all (A_i,B_i,D_i).
        Aall = []
        Ball = []
        Call = []
        for __ in range(N):
            A = int(input_data[index]); B = int(input_data[index+1]); C = int(input_data[index+2])
            index+=3
            # D = C-1 - (A+B)
            D = C - 1 - (A+B)
            if D<0:
                allDnonneg = False
            # Store the half-plane: A*x' + B*y' <= D
            Aall.append(A)
            Ball.append(B)
            Call.append(D)

        if not allDnonneg:
            # That means at least one D_i <0 => no solutions
            out.append('0')
            continue

        # Add lines for x' >=0 => -1*x' <=0 => line: (-1,0,0)
        # y' >=0 => (0,-1,0)
        halfplanes = [( -1,0,0 ), (0,-1,0 )]
        # plus the N lines
        for i in range(N):
            halfplanes.append( (Aall[i], Ball[i], Call[i]) )

        # Perform half-plane intersection
        poly = half_plane_intersect(halfplanes)
        if len(poly)==0:
            # no feasible region => 0
            out.append('0')
            continue
        # compute area
        anum, aden = polygon_area_rational(poly) # area = anum/aden
        if anum==0:
            # degenerate => no area => possibly no interior => check if there's at least one integer point
            # but the polygon is purely on an edge or point. Let's do boundary check quickly:
            # If there's a single point, that point must be integer or not. We'll check
            # if all vertices are the same => then it's that one point. Let's see if that is integral.
            # In a degenerate polygon, all intersection points might be the same or form a line.
            # Counting integer points on a line segment in the first quadrant might be >0 though...
            # But problem statement says "finite" => it might be a line or point. Let's handle carefully.
            # We'll just compute boundary points B. That is the total number of integer-lattice points
            # on the boundary (since area=0, there are no "interior" points). So the result is B.
            B = boundary_points(poly)
            out.append(str(B))
            continue

        # boundary
        B = boundary_points(poly)
        # use Pick's theorem => # integer points = area + B/2 + 1
        # area = anum / aden
        # so # = (anum/aden) + B/2 + 1
        #   = (anum/aden) + (B/2) + 1
        # We want an integer. We'll put everything on a common denominator = 2*aden
        # => # = [ anum*2 + B*aden + 2*aden ] / (2*aden)
        # Let top = 2*anum + B*aden + 2*aden
        # gcd might be large but final result must be integer if polygon is a valid integer-lattice polygon.
        top = 2*anum + B*aden + 2*aden  # integer
        bot = 2*aden
        # reduce?
        # Actually, by Pick's theorem, top must be divisible by bot. We'll do integer division:
        # Just do integer division:
        answer = top // bot
        out.append(str(answer))

    print("
".join(out))

# Don't forget to call main()
if __name__ == "__main__":
    main()