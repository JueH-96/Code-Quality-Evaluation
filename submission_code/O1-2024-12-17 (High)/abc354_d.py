def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    A, B, C, D = map(int, input_data)
    
    #----------------------------------------------------------------------
    # Observations and overall plan:
    #
    # 1) The plane is divided by lines:
    #    - x = integer
    #    - y = even-integer
    #    - x + y = even-integer
    #
    #    Any two adjacent regions (sharing a boundary line) differ in color.
    #    The region containing (0.5, 0.5) is black.
    #
    # 2) One can show the color of a point (x, y) is given by:
    #
    #       col(x, y) = ( floor(x)
    #                    + floor(y/2)
    #                    + floor((x+y)/2 )
    #                  ) mod 2
    #
    #    with (col = 0) denoting black, (col = 1) denoting white.
    #
    # 3) It turns out this coloring is 4×4-periodic in both x and y:
    #    shifting x or y by 4 does not change the color.  Consequently,
    #    exactly half the area of any full 4×4 block is black (area=16, black=8).
    #
    # 4) To find the black area in [A,C]×[B,D], we want an exact formula
    #    without enumerating all lines explicitly, as C−A, D−B can be up to 2e9.
    #
    # 5) We will use a 2D "prefix-sum" idea, but for a 2D periodic function f(x,y)
    #    = 1 if (x,y) is black, 0 if white, with period 4 in both directions.
    #
    #    Define F(X, Y) = area of black in the rectangle [0, X]×[0, Y], for X≥0,Y≥0.
    #    Then for any rectangle [x1,x2]×[y1,y2] with x1<x2,y1<y2 and x1,y1≥0:
    #
    #        area_black = F(x2, y2) - F(x1, y2) - F(x2, y1) + F(x1, y1).
    #
    #    We only need to implement F(X,Y) efficiently for all real X≥0,Y≥0.
    #
    # 6) Because of 4×4 periodicity, we write X = 4*M + x', with 0 ≤ x' < 4,
    #    Y = 4*N + y', with 0 ≤ y' < 4.  Then
    #
    #        F(X,Y) = M*N*(8)   # 8 = black area in one 4×4 block
    #                + M * G(4, y')    # leftover vertical strips
    #                + N * G(x', 4)    # leftover horizontal strips
    #                + G(x', y')       # leftover small rectangle
    #
    #    where G(x,y) = area of black in [0,x]×[0,y], restricted to the "fundamental" 4×4 cell.
    #    G(x,y) is at most 16 in area, so we can compute it exactly by subdividing
    #    the 4×4 block into up to 16 squares (each 1×1), some of which are further
    #    bisected by a diagonal (x+y= constant).  We label each sub-region black or white.
    #
    # 7) For negative A,B, we can exploit the periodicity.  We can shift the entire plane
    #    by (−4 * floor(A/4), −4 * floor(B/4)) so that the rectangle maps to a region
    #    [X1,X2]×[Y1,Y2] with X1,Y1 ≥ 0.  The color pattern does not change (shifting
    #    by multiples of 4 in x or y does not change the coloring).  Then we can apply
    #    the standard prefix-sum formula above on nonnegative coordinates.
    #
    # 8) We must output twice the black area.  We will do all big multiples in integer,
    #    and carefully handle the small leftover area in floating so we can do an
    #    exact rounding (it is at most 16, so floating round-off is safe).
    #
    #----------------------------------------------------------------------
    
    #=================== Utility: area of intersection with x+y <= K ===================
    def area_below_line_rect(lx, rx, ly, ry, K):
        """
        Returns area( { (x,y) in [lx,rx]×[ly,ry] : x+y <= K } ).
        All arguments are floats, but here in [0,4], so safe from large-floating issues.
        """
        if rx <= lx or ry <= ly:
            return 0.0
        
        # Let xA = K - ry, xB = K - ly.
        # For x < xA => y <= K-x >= K - xA = ry => entire vertical strip is inside.
        # For x > xB => y <= K-x <= K - xB = ly => empty.
        xA = K - ry  # boundary where line crosses y=ry
        xB = K - ly  # boundary where line crosses y=ly
        
        area_total = 0.0
        
        # region1: x in [lx, min(rx,xA)] => entire height
        x1_l = lx
        x1_r = min(rx, xA)
        if x1_r > x1_l:
            area_total += (x1_r - x1_l) * (ry - ly)
        
        # region2: x in [max(lx,xA), min(rx,xB)] => partial integration
        x2_l = max(lx, xA)
        x2_r = min(rx, xB)
        if x2_r > x2_l:
            # integrate ∫ from x2_l..x2_r of (K - x - ly)
            dx = x2_r - x2_l
            # integral of (K - ly - x) dx = (K - ly)*dx - (x^2/2) from x2_l..x2_r
            area_cut = ( (K - ly)*dx
                         - (x2_r*x2_r - x2_l*x2_l)*0.5 )
            if area_cut > 0:
                area_total += area_cut
        
        return area_total
    
    #=================== Color tests for squares ===================
    #
    # For squares (i, j) with i,j in {0,1,2,3}:
    #  - if i+j is even, that 1×1 square is a single color. We test the center's color.
    #  - if i+j is odd, the square is split by the diagonal x+y=i+j+1 into two triangles.
    #    We test col(i+0.25, j+0.25) to see if that "lower triangle" is black or white.
    #
    # col(x,y) = ( floor(x) + floor(y/2) + floor((x+y)/2 ) ) mod 2
    # We'll define small integer-based shortcuts for i, j.

    def color_square_center(i, j):
        """
        Returns 0 if black, 1 if white for the center of the square [i,i+1]×[j,j+1],
        used when i+j is even.
        
        We'll sample at (i+0.5, j+0.5).
        col(i+0.5, j+0.5) = ( i + floor((j+0.5)/2) + floor((i+j+1)/2 ) ) mod 2
        But let's just do it directly in Python integer math.
        """
        # floor(i+0.5) = i
        # floor((j+0.5)/2) = floor(j/2) because j is integer
        # floor((i+j+1)/2) = floor((i+j)/2 + 0.5) = (i+j)//2
        return (i + (j//2) + ((i+j)//2)) & 1  # mod 2
    
    def color_triangle_lower(i, j):
        """
        Returns the color (0=black,1=white) of the "lower triangle" in the square [i,i+1]×[j,j+1]
        when i+j is odd. We sample at (i+0.25, j+0.25).
        
        col(i+0.25, j+0.25) = ( i + floor((j+0.25)/2) + floor((i+j+0.5)/2 ) ) mod 2
        For integer i,j with i+j odd:
          - floor((j+0.25)/2) = j//2
          - floor((i+j+0.5)/2) = (i+j)//2  (since i+j is odd, i+j=2s+1 => floor((2s+1+0.5)/2)= s )
        Hence color = ( i + j//2 + (i+j)//2 ) mod 2
        """
        return (i + (j//2) + ((i+j)//2)) & 1
    
    #=================== G(x,y): area of black in [0,x]×[0,y], x,y in [0,4] ===================
    def G(x, y):
        """
        Compute (as a float) the area of black points in [0,x]×[0,y], with 0 <= x,y <= 4.
        We'll subdivide into squares i=0..3, j=0..3, then clip to [0,x]×[0,y].
        Each 1×1 square is either fully a single color (if i+j even) or
        is split by diagonal x+y=i+j+1 (if i+j odd).  We add up partial areas that are black.
        """
        area_sum = 0.0
        
        for i in range(4):
            if i >= x:  # no overlap in x
                break
            # The overlap in x-dimension with [i,i+1] is [Lx,Rx]
            Lx = float(i)
            Rx = float(i+1)
            if Rx > x:
                Rx = x
            
            if Rx <= Lx:
                continue
            
            for j in range(4):
                if j >= y:  # no overlap in y
                    break
                Ly = float(j)
                Ry = float(j+1)
                if Ry > y:
                    Ry = y
                if Ry <= Ly:
                    continue
                
                # Overlap region O = [Lx,Rx]×[Ly,Ry]
                w = (Rx - Lx)
                h = (Ry - Ly)
                area_ov = w*h
                
                s = i + j
                if (s & 1) == 0:
                    # i+j even => entire square has one color
                    c = color_square_center(i, j)  # 0 or 1
                    if c == 0:
                        # black
                        area_sum += area_ov
                    # else white => +0
                else:
                    # i+j odd => diagonal splits it
                    # T1: {x+y < i+j+1}, T2: {x+y > i+j+1}
                    # color of T1 is color_triangle_lower(i, j).
                    cT1 = color_triangle_lower(i, j)  # 0=black,1=white
                    # areaBelow = area(O ∩ {x+y <= i+j+1})
                    K = float(s + 1)  # i+j+1
                    areaBelow = area_below_line_rect(Lx, Rx, Ly, Ry, K)
                    if cT1 == 0:
                        # T1 is black => T2 is white
                        area_sum += areaBelow
                    else:
                        # T1 is white => T2 is black
                        area_sum += (area_ov - areaBelow)
        
        return area_sum
    
    #=================== getF2(X, Y): 2× area of black in [0,X]×[0,Y], with X,Y ≥0 ===================
    #
    # Because of the 4×4 periodic block, if X=4*M + x', Y=4*N + y',
    #   area = 8*M*N + M*G(4,y') + N*G(x',4) + G(x',y').
    # Then we multiply by 2 to keep an integer result.  We'll do large terms in integer
    # and small partial areas in float, rounding carefully.
    
    def getF2(X, Y):
        """
        Returns an integer = 2 * ( area of black in [0,X]×[0,Y] ).
        We define 0 if X<=0 or Y<=0.
        """
        if X <= 0 or Y <= 0:
            return 0
        
        # Floors
        M = int(X // 4)
        N = int(Y // 4)
        xprime = X - 4.0*M
        yprime = Y - 4.0*N
        
        # Big area part: 8*M*N => times 2 => 16*M*N
        big_part_2 = 16 * (M*N)  # can be very large, but fits in Python int
        
        # leftover
        # G(4, yprime), G(xprime, 4), G(xprime, yprime) are at most 16 in area.
        gy = G(4.0, yprime) if yprime>0 else 0.0
        gx = G(xprime, 4.0) if xprime>0 else 0.0
        gxy = G(xprime, yprime) if (xprime>0 and yprime>0) else 0.0
        
        leftover_area = M*gy + N*gx + gxy  # float <= M*16 + N*16 +16, but M,N could be large
        # but we only add leftover_area *2 to big_part_2. leftover_area <= (M+N+1)*16 in worst case
        # We risk losing the fractional part if we do big_part_2 + leftover_area*2 in float
        # So we do the integer part in big_part_2, and do round( leftover_area *2 ).
        
        leftover_twice = int(round(leftover_area * 2))
        
        return big_part_2 + leftover_twice
    
    #=================== areaRect2(X1, Y1, X2, Y2): 2× area of black in that rectangle =================
    def areaRect2(x1, y1, x2, y2):
        if x2 <= x1 or y2 <= y1:
            return 0
        return ( getF2(x2, y2)
               - getF2(x1, y2)
               - getF2(x2, y1)
               + getF2(x1, y1) )
    
    #----------------------------------------------------------------------
    # Now handle the input rectangle [A,C]×[B,D].  Possibly A,B <0.
    # We exploit 4×4 shift invariance: shiftX = -4*(floor(A/4)), shiftY = -4*(floor(B/4)).
    # Then A' = A + shiftX in [0,4), same for B'.  The region becomes:
    #   [A', C']×[B', D'] where C' = C + shiftX, D' = D + shiftY, and A'<C', B'<D'.
    # We then compute areaRect2(A', B', C', D').
    #
    # Finally we print that value (which is 2*(area)).
    #----------------------------------------------------------------------
    
    import math
    
    def floor_div_4(x):
        # In Python, x//4 is already floor-division even for negative.
        return x//4
    
    # Compute shifts:
    AX = A // 4  # floor_div_4(A)
    BX = B // 4
    
    shiftX = -4*AX
    shiftY = -4*BX
    
    A_prime = A + shiftX
    B_prime = B + shiftY
    C_prime = C + shiftX
    D_prime = D + shiftY
    
    # We want 2 x area of black in [A_prime, C_prime]×[B_prime, D_prime].
    # But ensure we pass them in ascending order to areaRect2:
    # A_prime < C_prime, B_prime < D_prime guaranteed by A<C, B<D.
    
    ans2 = areaRect2(float(A_prime), float(B_prime), float(C_prime), float(D_prime))
    
    # Output the result:
    print(ans2)

# Don't forget to call main().
if __name__ == "__main__":
    main()