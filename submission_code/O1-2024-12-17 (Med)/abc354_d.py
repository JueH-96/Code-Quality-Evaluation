def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    A, B, C, D = map(int, input_data)

    # ----------------------------------------------------------------
    # Explanation of the approach (high-level):
    #
    # 1) The plane is subdivided by three families of lines:
    #       x = integer
    #       y = even integer
    #       x + y = even integer
    #    and adjacent regions along these lines alternate colors.
    #    The region containing (0.5, 0.5) is black.
    #
    # 2) It turns out (and can be shown with careful case analysis)
    #    that, if you look at each "integral-cell" of width 1 in x
    #    and height 2 in y (i.e. [n, n+1] x [2m, 2m+2]), the black area
    #    in that 1x2 cell is always either 1.5 or 0.5, depending only
    #    on n mod 4.  In particular:
    #
    #       If n mod 4 is 0 or 1, then the black sub-area in
    #       that 1×2 cell is 1.5.
    #
    #       If n mod 4 is 2 or 3, then the black sub-area in
    #       that 1×2 cell is 0.5.
    #
    #    Moreover, this pattern does NOT depend on m mod anything.
    #    (The diagonal lines do subdivide each 1×2 strip, but after
    #    working out which sub-pieces are black/white, the total black
    #    area in the 1×2 block is fixed by n mod 4.)
    #
    # 3) Therefore, for y >= 0, the black area in the strip
    #       x in [n, n+1],  y in [0, 2K]
    #    is K times either 1.5 or 0.5 (depending on n mod 4).
    #    If y is not a multiple of 2, we have a leftover partial
    #    height < 2 at the top, which needs a small formula.  But we
    #    can work that out explicitly by again subdividing that top
    #    partial 1×(leftover_height) strip by the diagonal line(s).
    #
    # 4) We then build a function BlackAreaPos(x, y) that returns
    #    the black area within [0, x] × [0, y], for x, y >= 0.
    #    We do so by summing up whole 1×2 blocks plus the partial top
    #    strip in each vertical column, plus the partial rightmost
    #    column if x is not an integer.
    #
    # 5) For the general rectangle [A, B]×[C, D], we use the standard
    #    inclusion-exclusion:
    #
    #        BlackArea(A,B,C,D)
    #          = BlackAreaPos(C,D)
    #            - BlackAreaPos(A,D)
    #            - BlackAreaPos(C,B)
    #            + BlackAreaPos(A,B)
    #
    #    where BlackAreaPos() is clamped to zero if its arguments
    #    are negative.  (That is, if x<0 or y<0 in BlackAreaPos(x,y),
    #    we return 0.  This effectively splits the plane into
    #    four quadrants with corner at (0,0).)
    #
    # 6) Finally, we print 2 × (that black area).  By the problem
    #    statement, this should be an integer.
    #
    # The main nontrivial step is to implement BlackAreaPos(x, y)
    # carefully.  Below is a reasonably direct implementation.
    #
    # ----------------------------------------------------------------

    import math

    # Precompute how much black area each full 1×2 cell contributes,
    # depending on n mod 4:
    #   cellBlack[n mod 4] = black area in [n, n+1] x [2m, 2m+2]
    #                        (independent of m)
    cell_black = [1.5, 1.5, 0.5, 0.5]

    # We will also want a quick way to sum up cell_black from n=0..(x-1).
    # Observe that over each block of 4 consecutive n, the total is
    #   1.5 + 1.5 + 0.5 + 0.5 = 4.0
    # So we can do fast computation for large x.

    def black_area_in_full_strips(num_full_int_cols, num_full_2high):
        """
        Returns the black area in the region:
           x in [0, num_full_int_cols], y in [0, 2 * num_full_2high]
        where both arguments are nonnegative integers.

        This is simply the sum over n=0..(num_full_int_cols-1) of
          num_full_2high * cell_black[n mod 4].
        """
        if num_full_int_cols <= 0 or num_full_2high <= 0:
            return 0.0

        # Number of full 4-wide cycles in x-direction:
        cycles = num_full_int_cols // 4
        remainder = num_full_int_cols % 4

        # Sum for the cycles:
        # Each cycle (n mod4 = 0,1,2,3) sums to cell_black[0]+...+cell_black[3] = 4.0
        base_sum = cycles * 4.0

        # Sum for the leftover columns:
        leftover_sum = 0.0
        for r in range(remainder):
            leftover_sum += cell_black[r]

        total_per_2height = base_sum + leftover_sum  # black area for height=2
        return total_per_2height * num_full_2high

    def black_area_in_partial_height(n, y0, y1):
        """
        Black area in the strip x in [n, n+1], y in [y0, y1],
        where y1 - y0 <= 2 and y0,y1 within the same "2-high band".
        We assume 0 <= y1 - y0 <= 2.

        This function works by subdividing the 1×(h) region with the
        diagonal line x'+y' = 1 or 2, depending on (n + 2*m) mod2, etc.
        But we already know the total 1×2 block area results if the
        block was full [2m, 2m+2]. Here it is a partial sub-interval
        of length h = (y1 - y0).

        We'll do a small direct check by shifting the cell so that
        its bottom is at 0.  Then the line that might cut the cell
        is either x'+y' = 1 or x'+y' = 2 or no interior line if
        it's smaller than that.  We'll pick a test point to see
        which side is black vs white.  Then compute the intersection
        area.  However, to keep things simpler (and to avoid re-doing
        all diagonal logic here), we'll reuse the same "two-subregion"
        geometry we had for a full 1×2 block, then slice by height=h.
        """
        h = y1 - y0
        if h <= 0.0000000001:
            return 0.0
        if h >= 1.9999999999:
            # If effectively ~2 high, just return the full 1×2 cell's black area:
            return cell_black[n % 4]

        # Otherwise 0 < h < 2.  We'll do a small piece of the "cell" analysis.
        # Let e = (n + 2*m) mod 2, but we only need n mod 4 to know
        # how the full 1×2 was divided (since the black area phenomenon
        # for partial heights ended up depending just on n mod4).
        # We'll replicate the same shape we had: if n mod4 in {0,1},
        # the "diagonal" is x'+y'=2. If n mod4 in {2,3}, the "diagonal" is x'+y'=1
        # that enters the interior.  Actually from the previous analysis:
        #
        #  - n mod4 in {0,1} => the interior diagonal is x'+y'=2, which
        #    only becomes relevant if h >= 2. For 0 < h <2, we do NOT
        #    actually cut the region. So in that scenario, effectively
        #    the partial region is all part of one sub-piece or
        #    we might see the line in the top boundary if h=2 exactly.
        #
        #  - n mod4 in {2,3} => the interior diagonal that cuts the 1×2
        #    block is x'+y'=1. So if h>1, that line does subdivide the cell
        #    up to y'=h. We can then compute how much is below/above it.
        #
        # In short:
        #  * If n mod4 in {0,1}, the diagonal is at x'+y'=2. But if h<2,
        #    we never see the diagonal inside (0<x'<1,0<y'<h). So the partial
        #    region is a rectangle of area = 1*h, all in the same color
        #    as the region "below x'+y'=2" in the full block.  From the
        #    full-block analysis, the area below x'+y'=2 was (1.5) in a 1×2,
        #    so that subregion was black or white?  Actually that subregion
        #    had area=1.5, and it was black if n mod4 in {0,1} matched
        #    certain parity.  In fact from the table: n mod4 in {0,1}
        #    => the 1×2 block's black area is 1.5.  That means the "larger"
        #    portion of the block is black.  Indeed the region below x'+y'=2
        #    is 1.5 in area, so that is the black part.  So for 0 < h < 2,
        #    that entire partial region is inside that black subregion,
        #    giving black area = h * 1 (width).  Because it’s 100% black.
        #
        #  * If n mod4 in {2,3}, the interior diagonal is at x'+y'=1, which
        #    does cut the 1×2 block.  The smaller triangular area (size 0.5)
        #    is one color, and the bigger trapezoid (1.5) is the other color.
        #    From the table: if n mod4 in {2,3}, the total black area is 0.5.
        #    That means the smaller triangular region is black, the larger is white.
        #    The diagonal line from (0,1) to (1,0).
        #    So for a partial height h <1, we are entirely in that lower triangle,
        #    which is black.  Its area is (width=1) * h / 2, because it's the
        #    area under the line x'+y'=1.  So black area = 0.5*h.
        #    If h>1, then the region from y'=0 to y'=1 is that black triangle,
        #    plus from y'=1 to y'=h is above that line => that part is white.
        #    So if h>1, the black area is just the area of that lower triangle,
        #    which is 0.5 * 1^2 = 0.5, and does not grow beyond y'>1.
        #
        # Let's encode these rules.  All of these statements hold for an
        # interior strip (i.e. ignoring the possibility that crossing x=... is a boundary
        # on the left?), but that is exactly the scenario for the sub-strip
        # [n, n+1] × [y0, y1].  We only need the area portion.

        nm4 = n % 4

        if nm4 in (0, 1):
            # The partial 1×h region is entirely in the "below x'+y' = 2" portion,
            # which is black in that full-block analysis.  So black area = 1*h.
            return 1.0 * h
        else:
            # nm4 in {2,3}, the diagonal is x'+y' = 1 dividing black/white.
            # The lower triangle (0<x'<1, 0<y'<1, x'+y'<=1) is black, area=0.5.
            # If h <= 1, we only get a smaller triangle of height h, black area= (h^2)/2? Not quite:
            #   Actually, if we slice that lower triangle at y'=h <1, its boundary with x'=1-h.
            #   So area = 0.5*h if h <=1?  Let's be precise:
            #
            #   The line is x'+y'=1 => x' in [0, 1-y'] for 0<=y'<=1.  If h<1,
            #   the region is 0<=y'<=h, x'<=1-y'.  The area is ∫(y'=0..h) of (1-y') dy'
            #                                         = ∫(0..h) 1 dy' - ∫(0..h) y' dy'
            #                                         = h - h^2/2
            #   That is h - h^2/2 = h(1 - h/2).  If h=1, that is 0.5.  If h=0.5, that is 0.5-0.125=0.375, etc.
            #
            #   So for 0<= h <=1, black area = h - h^2/2.
            #
            #   If h >=1, then the black portion is the entire lower triangle (area=0.5),
            #   plus the region from y'=1..h is the upper trapezoid, which is white. So total black=0.5.
            #
            if h <= 1:
                return h - (h*h)/2.0  # = ∫(0..h)(1-y')dy'
            else:
                return 0.5

    def black_area_in_strip(n, height):
        """
        Black area in the strip x in [n, n+1], y in [0, height], with height >= 0.
        We'll break y into full chunks of 2 plus a leftover <2, and sum up.
        """
        if height <= 0:
            return 0.0
        # Number of full 2-high blocks:
        full_blocks = int(height // 2)
        leftover = height - 2 * full_blocks
        # Base from full blocks:
        base = full_blocks * cell_black[n % 4]
        # Add partial leftover:
        # that's the region x in [n,n+1], y in [2*full_blocks, 2*full_blocks + leftover]
        part = black_area_in_partial_height(n, 2*full_blocks, 2*full_blocks + leftover)
        return base + part

    def black_area_pos(x, y):
        """
        Black area in [0, x] × [0, y], clamped to 0 if x<=0 or y<=0.
        We'll sum integer columns [n, n+1] up to floor(x), then
        add the partial column if there's a fractional part in x.
        """
        if x <= 0 or y <= 0:
            return 0.0
        # Floor of x:
        Nx = int(math.floor(x))
        rx = x - Nx  # leftover fraction in x, 0 <= rx < 1

        # Sum over full integer columns:
        total = 0.0
        for n in range(Nx):
            total += black_area_in_strip(n, y)

        # Now add the partial column [Nx, Nx+rx], if rx>0
        if rx > 1e-14:
            # In principle, we subdivide it by y=2m lines.  But simpler is:
            # we treat it as "fractional width" times the same y-partitioning.
            # However, that won't quite work directly, because crossing x= Nx + something
            # is indeed the same 'strip index' n.  So we can scale the black area
            # in that strip by (rx / 1.0)?  Actually, not so fast: the pattern flips
            # on the line x = Nx+1, but we never reach that in fractional width.
            #
            # So effectively, the color arrangement in [Nx, Nx+1] is the same as
            # we used, but now the width is rx instead of 1.  The diagonal lines
            # inside that 1×2 cell geometry are scaled in x'.  So the black
            # sub-area scales linearly with the fraction rx (since the lines
            # are all of the form x'+y' = const).  So we can do:
            #   full_strip_black = black_area_in_strip(Nx, y)   (that assumes full width=1)
            #   scaled by rx => partial_black = full_strip_black * rx
            #
            # This works because within that unit cell, every horizontal cross-section
            # from y to y+dy has black/white partition in x' from 0..some boundary.
            # If we only allow x' up to rx (instead of 1.0), the black portion
            # of that cross-section is scaled by rx as well.  The diagonal x'+y'=k
            # remains a straight line in that scaled domain.  In short, the portion
            # of area is linear in the width fraction for 0 <= x' <= rx (the pattern
            # is "affinely" scaled).
            #
            # So we do:
            full_strip_black = black_area_in_strip(Nx, y)
            partial_black = full_strip_black * rx
            total += partial_black

        return total

    def area_black_in_rectangle(x1, y1, x2, y2):
        """
        Returns the black area in [x1, x2] × [y1, y2].
        Uses inclusion-exclusion with black_area_pos() defined for [0,x]×[0,y].
        """
        # If the rectangle is empty or invalid, return 0
        if x2 <= x1 or y2 <= y1:
            return 0.0

        # We define a helper: BAPos(x, y) = black_area_pos(max(0, x), max(0, y))
        # i.e. clamp to 0 if negative
        def BAPos(x, y):
            return black_area_pos(max(0, x), max(0, y))

        # Inclusion-exclusion:
        # B([x1,x2]×[y1,y2]) = B([0,x2]×[0,y2]) - B([0,x1]×[0,y2])
        #                     - B([0,x2]×[0,y1]) + B([0,x1]×[0,y1])
        return (
            BAPos(x2, y2)
            - BAPos(x1, y2)
            - BAPos(x2, y1)
            + BAPos(x1, y1)
        )

    # Compute the black area in [A,C] × [B,D], then print 2× that area.
    black_area = area_black_in_rectangle(A, B, C, D)
    # The problem guarantees that 2*area is an integer, but it can be very large.
    ans = 2 * black_area

    # Print as an integer
    # (avoid tiny floating rounding by rounding to nearest int)
    print(int(round(ans)))