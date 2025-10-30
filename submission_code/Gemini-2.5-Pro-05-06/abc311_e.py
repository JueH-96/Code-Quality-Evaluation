import sys

def solve():
    H, W, N = map(int, sys.stdin.readline().split())

    holed_squares = set()
    for _ in range(N):
        r_hole, c_hole = map(int, sys.stdin.readline().split())
        holed_squares.add((r_hole, c_hole))

    # dp_next corresponds to values for row r+1.
    # dp_curr corresponds to values for the current row r.
    # Both lists are of size W+2. Indices 0 and W+1 are padding (always 0),
    # simplifying boundary checks in the min() function.
    # Grid cells (r,c) correspond to dp_curr[c] or dp_next[c].
    
    # Initialize dp_next for row H+1 (which is outside the grid, so all values are 0).
    dp_next = [0] * (W + 2) 
    total_holeless_squares = 0

    # Iterate rows from bottom to top (r from H down to 1).
    for r in range(H, 0, -1):
        # Initialize dp_curr for the current row r.
        # Its end elements dp_curr[0] and dp_curr[W+1] will remain 0 due to initialization.
        dp_curr = [0] * (W + 2)
        
        # Iterate columns from right to left (c from W down to 1).
        for c in range(W, 0, -1):
            if (r, c) in holed_squares:
                # If current cell (r,c) is holed, no square can start here.
                dp_curr[c] = 0
            else:
                # If (r,c) is not holed, it can at least form a 1x1 square.
                # The side length of the largest square starting at (r,c) is
                # 1 + minimum of side lengths of squares starting at:
                #   (r, c+1)   -- value is dp_curr[c+1] (already computed for this row r)
                #   (r+1, c)   -- value is dp_next[c]   (from previous row r+1 iteration)
                #   (r+1, c+1) -- value is dp_next[c+1] (from previous row r+1 iteration)
                dp_curr[c] = 1 + min(dp_curr[c+1], dp_next[c], dp_next[c+1])
            
            # Each dp_curr[c] value (i.e., max side length of square starting at (r,c))
            # also represents the count of holeless squares whose top-left corner is (r,c).
            # (Squares of side 1, 2, ..., up to dp_curr[c]).
            total_holeless_squares += dp_curr[c]
        
        # After computing all values for row r, this dp_curr (now representing row r)
        # becomes dp_next for the next iteration (which will compute row r-1).
        dp_next = dp_curr

    print(total_holeless_squares)

if __name__ == '__main__':
    solve()