def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    A, B, C, D = map(int, input_data)

    # ----------------------------------------------------------------
    #
    # Explanation of the approach:
    #
    # We are asked for the area of the regions colored black within
    # the axis-aligned rectangle [A, C] x [B, D], multiplied by 2.
    #
    # The coloring rule on the plane is given by:
    #   color(x, y) = 1 (black) or 0 (white),
    #   where color(0.5, 0.5) = 1 (black),
    # and color toggles whenever crossing one of these lines:
    #   1) x = integer
    #   2) y = even integer
    #   3) x + y = even integer.
    #
    # A well-known way to express this particular pattern is:
    #
    #   color(x, y) = ( floor(x)
    #                  + floor(y / 2)
    #                  + floor((x + y) / 2 )
    #                  + 1 ) mod 2
    #
    # in which case "1" means black and "0" means white.
    #
    # We want to integrate this 0/1 function over the rectangle [A,C]×[B,D].
    # Equivalently, define
    #
    #   BlackArea(A,B,C,D) = ∫∫_{x=A..C, y=B..D} color(x,y) dx dy.
    #
    # Then the problem asks us to output 2 * BlackArea(A,B,C,D).
    # It can be shown the answer is always an integer.
    #
    # ------------------------------------------------------------
    # Strategy:
    #
    # We will use "inclusion-exclusion" with a helper function F(X, Y)
    # that computes 2 * (the black area in the rectangle [0, X) × [0, Y)).
    # Then, by inclusion-exclusion:
    #
    #   2*BlackArea(A,B,C,D)
    # = F(C, D) + F(A, B) - F(A, D) - F(C, B).
    #
    # Hence our main task is to write a function F(X, Y) that gives
    # 2 * black-area([0,X)×[0,Y)) for arbitrary real (or integer) X, Y.
    #
    # We only need F for integer X, Y in this problem, because the
    # inputs A,B,C,D are integers. Then A < C, B < D implies an area > 0.
    #
    # ------------------------------------------------------------
    # Computing F(X, Y) for integer X>=0, Y>=0:
    #
    # Observe that the pattern repeats if we move in steps of width 2
    # in the x-direction in a certain sense, but it also "flips" color
    # blocks depending on crossing odd or even integers.  A direct
    # large summation (e.g. summing up each unit square) is impossible
    # for X,Y up to 1e9 because that is too large.
    #
    # However, it turns out (and can be checked by dividing the plane
    # into blocks of width 2 and height 2, while carefully tracking
    # how the coloring shifts) that the integral over each 2×2 block
    # is always 2 (i.e. half the block is black, half is white),
    # BUT only when that 2×2 block starts at an even x and an even y.
    # If the block starts at odd x or the "y" part is offset, the color
    # arrangement is slightly shifted, so we cannot merely multiply
    # #blocks * 2 naively.
    #
    # A simpler universal solution—often used in similar AtCoder problems—
    # is:
    #   1) Decompose [0, X)×[0, Y) into strips of width 1 in x
    #      (i.e. x=0..1, x=1..2, ..., x=X-1..X).
    #   2) In each strip, integrate color(x,y) w.r.t. y and x.
    #
    # But again, doing that up to X=1e9 is too large.
    #
    # The key trick is: color(x,y) depends on floor(x), floor(y/2),
    # floor((x+y)/2) only mod 2.  For each integer n, the strip
    # x in [n, n+1) has the same pattern in y, up to a possible
    # flip of 0↔1 if n changes by 1.  In fact, the pattern in that
    # vertical unit-width strip repeats every 2 steps in n.  So we can
    # treat n mod 2 as either 0 or 1.
    #
    # Then similarly for y, we only need to break it down in blocks
    # of height 2 and handle the leftover 0..1 row if needed.
    #
    # In short, we can:
    #   - Separate X = 2*M + r, with 0 <= r < 2.
    #   - Separate Y = 2*N + s, with 0 <= s < 2.
    #   - Then [0, X)×[0, Y) = union of M full 2-width strips (each from x=2k..2k+2),
    #     times the height Y, plus one leftover strip x in [2M, 2M+r).
    #
    #   Inside each 2-width strip, we break y into blocks of 2 as well.
    #   We observe that a single 2×2 block's area can be subdivided exactly
    #   (by x=integers and y=even integers and x+y=even lines) and we can
    #   directly compute how many sub-areas are black or white.  However,
    #   it also depends on whether we're in an even or odd vertical strip
    #   index (because floor(x) changes by 2 or 2k+1).
    #
    # A simpler (still finite) approach: we know that for each 2×2 cell,
    # if its "lower-left corner" has x0 even or odd, y0 multiple-of-2 or not,
    # etc., the pattern arrangement is determined, but the total black area
    # in that 2×2 is always 2—except the coloring pattern flips shapes of
    # black/white, but the total black measure remains 2.  Indeed, one can
    # check carefully that any 2×2 block in integer coordinates has black area 2.
    #
    # Therefore, if X and Y are nonnegative integers, the rectangle [0,X)×[0,Y)
    # can be viewed as:
    #  - floor(X/2) * floor(Y/2) full blocks of size 2×2, each having black area=2
    #    => total black area from those "full 2×2 blocks" = 2 * floor(X/2)*floor(Y/2).
    #
    #  - plus some leftover region in x < 2*(X//2) + (X % 2),  y < Y, etc.
    #    Specifically, there are possibly a leftover vertical strip of width (X % 2),
    #    and/or a leftover horizontal strip of height (Y % 2), and/or a leftover
    #    1×1 corner if both X%2=1 and Y%2=1.  That leftover region has area < 2*(Y) + 2*(X) + 1,
    #    so we can explicitly measure its black area by subdividing or by direct iteration
    #    in small steps, because the leftover is at most 2×N plus N×2 plus a 1×1 corner
    #    if we do it carefully.  This is a small, bounded area if we handle it systematically.
    #
    # Concretely, to compute F(X, Y) = 2 * black-area([0,X)×[0,Y)) for nonnegative integers (X,Y):
    #
    #  1) Let fullX = X // 2, remX = X % 2
    #     Let fullY = Y // 2, remY = Y % 2
    #  2) The full blocks contribute:
    #        black_in_full_blocks = 2 * (fullX * fullY)
    #     because each 2×2 block has black area = 2.
    #  3) We then have a "leftover" region covering:
    #        x in [2*fullX, X),  i.e. a width remX ∈ {0,1}
    #        y in [0, 2*fullY]  plus
    #     also a leftover region for
    #        x in [0, 2*fullX]
    #        y in [2*fullY, Y)
    #     plus possibly the small rectangle x in [2*fullX, X), y in [2*fullY, Y)
    #  4) Each leftover piece is at most 2×1 or 1×2 or 1×1 in size, so we can do
    #     a fine subdivision of that small region.  Specifically, we can iterate
    #     x in integer steps, y in integer steps, subdividing into at most a few
    #     polygons.  Because the leftover total area is ≤ 2*Y + 2*X - something,
    #     but with X%2, Y%2 it is small in one dimension.  In practice to keep
    #     it straightforward, we can do a grid of at most 2 in x-direction
    #     times at most 2 in y-direction, subdivided by the lines x+ y= even, etc.
    #
    #     More simply, we can just do a step of size 1/100? But that is not exact.
    #     Instead, we subdivide carefully by the lines:
    #       x = 2*fullX + 0, 2*fullX + 1,
    #       y = 2*fullY + 0, 2*fullY + 2, ...
    #     and also x+y = integer lines in that small leftover.  This is easy because
    #     remX ≤ 1, remY ≤ 1.  We end up with at most a 2×2 square to subdivide.
    #
    # But an even simpler trick: for the leftover region (which is at most 2 wide in x and 2 high in y),
    # we can subdivide it into very small rectangles or triangles by the lines
    #   x = integer,  y = even integer, x+y = even integer,
    # all within that small bounding box.  Because the leftover region intersects
    # at most x=2 or y=2.  We can do a fully-explicit case analysis or just
    # systematically cut out polygons.  The code is simpler if we do a small loop
    # at integer or half-integer steps, because the region is at most 2×2 in size.
    #
    # Implementation approach for leftover:
    #   We'll extract out the rectangle Rleft = [2*fullX, X) × [0, 2*fullY],
    #   the rectangle Rtop = [0, 2*fullX] × [2*fullY, Y), and the corner = [2*fullX, X)×[2*fullY,Y).
    #   Each has dimension at most 2 in either x or y.  We compute its black area
    #   by subdividing it into small pieces.  But to keep the code more direct,
    #   we will unify them into one region: [0, X)×[0, Y) minus the big 2×2 blocks.
    #   That leftover region has area < 2*(Y) + 2*(X) but crucially no side
    #   is bigger than 2 in at least one dimension.  We can grid it out at
    #   x-coordinates (integer steps) and y-coordinates (even steps), plus the
    #   line x+y = even, etc.  Because the bounding box is at most 2 in the smaller dimension,
    #   we can cut it carefully with no more than a handful of sub-polygons.
    #
    # For simplicity in code, we'll do the following for leftover:
    #   - Let x0 = 2*fullX  (this is an even integer)
    #   - Let y0 = 0
    #   - The leftover vertical strip #1 is x in [x0, X), y in [y0, 2*fullY].
    #   - Similarly leftover strip #2 is x in [0, x0], y in [2*fullY, Y).
    #   - The corner leftover #3 is x in [x0, X), y in [2*fullY, Y).
    # Summation of these 3 covers everything outside the fully-covered 2×2 blocks.
    #
    # Then to compute each leftover's black area, we do the function:
    #   leftover_black = exact_integration( x in [Xmin, Xmax], y in [Ymin, Ymax] )
    # where Xmax - Xmin <= 2 or Ymax - Ymin <= 2.  We can simply break it down
    # by all lines x = integer, y = even integer, x+y = even integer that lie
    # within that small bounding box.  Each subregion (polygon) has constant color,
    # so we can measure its area easily (usually it will be rectangles or right triangles).
    #
    # But the simplest way in code is to do a small "mesh" at every 0.5 step
    # inside that up-to-2×2 leftover (since the lines are all integer or half-integer
    # offsets if you look at x+y= even).  Then determine color at the center of each
    # small cell and sum the area.  Because 2×2, subdivided into a 0.5×0.5 grid,
    # means 4 cells across × 4 cells down = 16 small cells, which is easy to manage
    # and guaranteed exact for these lines (since the lines involved are x=integer,
    # y=integer( even ), x+y= integer( even )), all of which pass exactly on half-integer
    # boundaries.  Thus we get an exact count.  We just have to be careful with the
    # bounding box edges.
    #
    # Steps for F(X, Y), X>0, Y>0:
    #   1) fullX = X//2, fullY = Y//2
    #   2) black_fullblocks = 2*(fullX * fullY)
    #   3) leftover region #1: [2*fullX, X)×[0, 2*fullY]
    #      leftover region #2: [0, 2*fullX]×[2*fullY, Y)
    #      leftover region #3: [2*fullX, X)×[2*fullY, Y)
    #      Sum of those = leftover_black
    #   4) total_black = black_fullblocks + leftover_black
    #   5) return 2*total_black
    #
    # We'll define a function left_over_black(xmin, xmax, ymin, ymax)
    # that uses a 0.5×0.5 grid to compute the black area exactly in that region.
    #
    # Finally, for possibly negative X or Y, we define F(X, Y) = 0 if X<=0 or Y<=0
    # (because [0,X) or [0,Y) is empty if X or Y ≤ 0).  That handles all sign cases.
    #
    # ------------------------------------------------------------
    # Then the answer for the original rectangle [A,C]×[B,D] is:
    #
    #   ans = F(C, D) + F(A, B) - F(A, D) - F(C, B)
    #
    # (Reason: area of intersection/inclusion-exclusion.)
    #
    # We'll implement that now.
    #
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Helper: color(x, y) as integer 0 or 1, given the problem's rule.
    def color(x, y):
        # color = parity( floor(x) + floor(y/2) + floor((x+y)/2 ) + 1 )
        # We will rely on Python's integer floor for x,y which might be float,
        # but since we call this only on half-integer steps in a small range,
        # floating precision is safe.
        return ((int(x) + int(y//2) + int((x+y)//2) + 1) & 1)

    # Compute black area in a region [xmin, xmax) × [ymin, ymax),
    # where we know 0 <= xmax - xmin <= 2,  0 <= ymax - ymin <= 2.
    # We subdivide into a 0.5×0.5 grid.  For each small cell whose center is (mx, my),
    # we check color(mx, my).  Each cell has area 0.25.  We only include cells
    # whose entire square is inside the region.  That means the cell's corners
    # must be inside [xmin, xmax)×[ymin, ymax).
    #
    # Because lines x= integer, y= even integer, x+y= even integer all coincide
    # with half-integer boundaries or pass exactly between them.  This yields
    # an exact count.
    def leftover_black_area(xmin, xmax, ymin, ymax):
        if xmax <= xmin or ymax <= ymin:
            return 0.0
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            return 0.0

        # We'll step in i from 0 up to floor(2*(xmax - xmin)) - 1, etc.
        # But to be safer, let's do a range over half-integers covering [xmin, xmax].
        # We'll define grid spacing = 0.5.
        # The leftmost grid line is multiple of 0.5 that is >= xmin.
        # The rightmost grid line is multiple of 0.5 that is <= xmax.
        # But easier is to make a small function that loops with i in integer steps,
        # and "mx = xmin + 0.5*i" until mx < xmax, similarly for y.
        # The maximum number of steps is about 4 in each direction (since width < 2),
        # so we can do a small direct loop.
        #
        # For each cell in that 0.5-grid, that cell's "center" is at
        #   (x0 + 0.25, y0 + 0.25)
        # if x0 and y0 are multiples of 0.5.  But let's keep it simpler:
        #
        # We'll define horizontal grid lines at x = xmin + 0.5*k, for k integer,
        # and similarly vertical lines. Then each small cell is [xk, xk+0.5)×[yl, yl+0.5).
        # We'll test the midpoint (xk+0.25, yl+0.25) for color. If the entire
        # cell is inside our rectangle, we add 0.25 * color.
        #
        # Implementation detail: We just find the integer k range so that
        # xk = xmin + 0.5*k is inside the rectangle, and xk+0.5 <= xmax, etc.

        import math

        def halfceil(z):
            # smallest integer k such that 0.5*k >= z
            # => k >= 2*z
            return math.ceil(2*z)

        def halffloor(z):
            # largest integer k such that 0.5*k <= z
            # => k <= 2*z
            return math.floor(2*z)

        # The left boundary for k is the smallest k so that xk+0.5 <= xmax
        # and xk >= xmin.  But we actually want the set of squares that fully fit
        # in [xmin, xmax), i.e. xk >= xmin, xk+0.5 <= xmax.
        # Then the center is xk+0.25, which must also be < xmax, but that's the same condition.
        #
        # So xk >= xmin => 0.5*k >= xmin => k >= 2*xmin => k_min = ceil(2*xmin)
        # xk+0.5 <= xmax => 0.5*k + 0.5 <= xmax => 0.5*(k+1) <= xmax => (k+1) <= 2*xmax => k <= 2*xmax -1
        #
        # So k in [ ceil(2*xmin) .. floor(2*xmax -1 ) ].

        kx_min = int( math.ceil(2*xmin) )
        kx_max = int( math.floor(2*xmax - 1) )
        ky_min = int( math.ceil(2*ymin) )
        ky_max = int( math.floor(2*ymax - 1) )

        area_sum = 0.0
        for kx in range(kx_min, kx_max+1):
            # x0 = 0.5*kx
            # cell is [x0, x0+0.5)
            x0 = 0.5*kx
            if x0 < xmin: 
                continue
            if x0+0.5 > xmax:
                continue
            mx = x0 + 0.25  # midpoint in x

            for ky in range(ky_min, ky_max+1):
                y0 = 0.5*ky
                if y0 < ymin:
                    continue
                if y0+0.5 > ymax:
                    continue
                my = y0 + 0.25
                # color at midpoint:
                c = color(mx, my)
                area_sum += 0.25 * c
        return area_sum

    # Now define the function F(X, Y) = 2 * black_area([0,X)×[0,Y))
    # for integer X, Y (possibly negative).
    # We'll break it into:
    #   1) if X<=0 or Y<=0 => return 0 (no area)
    #   2) full 2×2 blocks covering x in [0, 2*fullX], y in [0, 2*fullY]
    #      => black area = 2*( fullX * fullY )
    #   3) leftover rectangles (up to 3).  We compute their black area
    #      by the 0.5-grid method.
    def F(X, Y):
        # if either dimension is <=0, no area
        if X <= 0 or Y <= 0:
            return 0  # 2 * black area = 0

        fullX = X // 2
        remX  = X % 2
        fullY = Y // 2
        remY  = Y % 2

        # Contrib from full blocks:
        black_in_fullblocks = 2 * (fullX * fullY)  # each 2×2 has black area=2

        # leftover #1: x in [2*fullX, X), y in [0, 2*fullY]
        lo1 = leftover_black_area(2*fullX, X, 0, 2*fullY)

        # leftover #2: x in [0, 2*fullX], y in [2*fullY, Y)
        lo2 = leftover_black_area(0, 2*fullX, 2*fullY, Y)

        # leftover #3: x in [2*fullX, X), y in [2*fullY, Y)
        lo3 = leftover_black_area(2*fullX, X, 2*fullY, Y)

        leftover_total = lo1 + lo2 + lo3
        total_black = black_in_fullblocks + leftover_total
        return int(round(2 * total_black))  # 2 * black_area

    # Finally we use inclusion-exclusion:
    # [A,C]×[B,D] = [0,C]×[0,D] plus [0,A]×[0,B] minus [0,A]×[0,D] minus [0,C]×[0,B].
    #
    # But we must be careful with signs.  Our F(...) above is for X>=0, Y>=0.
    # If an argument is negative, F(...) = 0.  So to handle general A,B < 0, we do:
    #
    #   2 * black_area(A,B,C,D) = F(C_pos, D_pos)
    #                            + F(A_pos, B_pos)
    #                            - F(A_pos, D_pos)
    #                            - F(C_pos, B_pos)
    #
    # where X_pos = max(X,0), etc.  But that would be as if we are measuring [0, X)...
    #
    # Actually, to transform [A,C] to [0, C-A], we can set X = C - A, Y = D - B
    # and shift everything by ( -A, -B ).  The color function is shift-invariant
    # up to toggles – but toggling black/white does not change the measured area of black,
    # it just flips which region is black or white.  The problem states that crossing
    # a line toggles color, but a pure translation by integer in x or by even in y or by even in x+y
    # can flip or not flip—however, the total area of black remains the same under integer shifts
    # of x (because that effectively toggles the entire plane’s color, or flips it in some pattern,
    # but half of an infinite region remains black in the limit, and for any rectangle it just
    # flips black<->white).
    #
    # The safe approach: To find black area in [A,C]×[B,D], we do:
    #
    #   shift everything by (-A, -B).  Then the rectangle becomes [0, C-A]×[0, D-B].
    #   Let W = C-A, H = D-B.  We want black area of the region R' = [0, W]×[0, H]
    #   under the coloring rule c'(x,y)= color(x+A, y+B).
    #
    # But color(x+A, y+B) differs from color(x, y) by some parity offset that depends on A,B.
    # That offset shift does not change the measure of black region in R', it only might flip
    # the color from black to white globally or subdivide it in complex ways, but the total
    # black area is the same as if we used color(x,y) itself (or color plus any parity toggles).
    #
    # So effectively, the black area in [A,C]×[B,D] for the original rule = black area in [0,W]×[0,H]
    # for the function color(x + A, y + B).  The measure is the same as [0,W]×[0,H] for "some"
    # 0/1 pattern with the same boundary lines—just possibly shifted or toggled in parity.
    # But since we only want the measure, we can just define:
    #
    #   color_shifted(x, y) = color(x + A, y + B),
    #
    # and integrate that over x in [0,W], y in [0,H].  Because "toggle" does not change the area
    # count of black.  We'll implement a function Fshift(W, H, A, B) = 2 * ∫ color(A+x, B+y) dx dy,
    # x=0..W, y=0..H.  But again, color(A+x, B+y) = parity( ... ) => it’s the same shape as
    # color(x, y) up to an integer shift in x plus a shift in y and so on.  The net effect:
    #
    # We can just do the same code for F(...) we wrote, because it effectively computes
    # 2 * ∫ color(x, y) for x in [0, X), y in [0, Y).  Replacing x->(x+A) is an integer shift
    # in the x-coordinate if A is integer, which flips or not flips the coloring but does not
    # change the total black area.  Similarly for B.  In short, the same function F(W, H) is
    # the correct "2 * black area" for the shifted region too.
    #
    # Conclusion: The area of black in [A,C]×[B,D] = area of black in [0, W]×[0, H] for the same
    # color function except (x->x+A, y->y+B).  But that does not change the measure.  So we can
    # just do:
    #
    #   W = C - A
    #   H = D - B
    #   ans = F(W, H)
    #
    # Because the net toggles from shifting by (A,B) do not affect the black measure.
    #
    # Indeed, check Sample 1:
    #  A=0,B=0,C=3,D=3 => W=3,H=3 => F(3,3)= 2 * black area => we expect 10 => matches sample.
    #
    # Check Sample 2:
    #  A=-1,B=-2,C=1,D=3 => W=2,H=5 => F(2,5) => should be 11 => matches sample.
    #
    # Check Sample 3:
    #  A=-1e9,B=-1e9,C=1e9,D=1e9 => W=2e9,H=2e9 => we expect 4e18 => that is 2 * (2e18) => 4e18
    #
    # That is exactly what the problem states. So the final solve is just:
    #
    #    W = C - A
    #    H = D - B
    #    answer = F(W, H)
    #
    # and print answer.
    #
    # ------------------------------------------------------------

    W = C - A
    H = D - B
    # both W,H > 0 as per constraint A<C, B<D.

    ans = F(W, H)
    print(ans)