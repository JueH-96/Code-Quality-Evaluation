# YOUR CODE HERE
import sys

def main():
    """
    Solves the AtCoder's Wallpaper problem.
    """
    A, B, C, D = map(int, sys.stdin.readline().split())

    # Precomputed table for 2 * black_area in [0, x] x [0, y]
    # for small x, y, based on the fundamental 4x2 cell of the pattern.
    # table[y][x] stores the value for a rectangle of size x by y.
    # y is in {0, 1, 2} and x is in {0, 1, 2, 3, 4}.
    # The values are derived by analyzing the coloring function
    # `(floor(x) + floor(y/2) + floor((x+y)/2)) % 2`.
    
    # y\x | 0 | 1 | 2 | 3 | 4
    # --- |---|---|---|---|---
    # 0   | 0 | 0 | 0 | 0 | 0
    # 1   | 0 | 2 | 3 | 3 | 4
    # 2   | 0 | 3 | 6 | 7 | 8
    
    table = [
        [0, 0, 0, 0, 0],
        [0, 2, 3, 3, 4],
        [0, 3, 6, 7, 8]
    ]

    def calc(X, Y):
        """
        Calculates 2 * black_area in the rectangle [0, X] x [0, Y] for non-negative X, Y.
        The calculation leverages the 4x2 periodicity of the wallpaper pattern.
        """
        if X < 0 or Y < 0:
            # This function is designed for non-negative coordinates.
            return 0
        
        # Decompose X and Y based on the 4x2 period.
        qX, rX = X // 4, X % 4
        qY, rY = Y // 2, Y % 2

        # The total 2 * area is composed of four parts:
        # 1. Full 4x2 tiles: qX * qY of them. Each has 2 * black_area = 8.
        # 2. A strip on the right: size rX x 2*qY.
        # 3. A strip on the top:   size 4*qX x rY.
        # 4. A corner rectangle:  size rX x rY.
        
        # Because of periodicity p(x+4k, y+2l) = p(x,y), we can analyze these parts
        # by shifting them to the origin and using the precomputed table.

        # 1. Full tiles
        res = qX * qY * 8
        
        # 2. Right strip (qY copies of [0, rX] x [0, 2])
        res += qY * table[2][rX]

        # 3. Top strip (qX copies of [0, 4] x [0, rY])
        res += qX * table[rY][4]
        
        # 4. Corner rectangle ([0, rX] x [0, rY])
        res += table[rY][rX]
        
        return res

    # The coloring pattern p(x,y) is invariant under shifts (x+4k, y+2l).
    # This means the black area inside a rectangle is invariant under such shifts.
    # We choose a shift S (a multiple of 4, and thus 2) large enough to make
    # all coordinates non-negative.
    
    SHIFT = 4 * 10**9

    # Apply the principle of inclusion-exclusion to the shifted coordinates.
    # Area(Rect) = Area(0,C,0,D) - Area(0,A,0,D) - Area(0,C,0,B) + Area(0,A,0,B)
    
    ans = (calc(C + SHIFT, D + SHIFT) - 
           calc(A + SHIFT, D + SHIFT) - 
           calc(C + SHIFT, B + SHIFT) + 
           calc(A + SHIFT, B + SHIFT))
           
    print(ans)

if __name__ == "__main__":
    main()