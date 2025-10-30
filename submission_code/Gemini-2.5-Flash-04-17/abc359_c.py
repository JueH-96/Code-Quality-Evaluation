import sys

def solve():
    sx, sy = map(int, sys.stdin.readline().split())
    tx, ty = map(int, sys.stdin.readline().split())

    # The minimum toll is the minimum number of tile boundaries crossed
    # when moving from (sx+0.5, sy+0.5) to (tx+0.5, ty+0.5).
    # The coordinate plane is tiled by rectangles [bx, bx+2] x [by, by+1]
    # where bx + by is even.
    # A point (x, y) is in tile [bx, bx+2] x [by, by+1] if
    # by = floor(y)
    # bx = floor(x) if floor(x) + floor(y) is even
    # bx = floor(x) - 1 if floor(x) + floor(y) is odd

    # A move crosses a horizontal boundary y = j (integer).
    # Toll is always paid when crossing a horizontal grid line y=j.
    # The number of horizontal grid lines crossed is |ty - sy|.
    # Total vertical toll is |ty - sy|.
    vertical_toll = abs(ty - sy)

    # A move crosses a vertical boundary x = i (integer).
    # Toll is paid if i + floor(y) is even at the crossing point.
    # The path must cross vertical grid lines x=i for i from min(sx, tx)+1 to max(sx, tx).
    # The number of vertical grid lines crossed is |tx - sx|.
    # Horizontal toll is the sum of tolls for crossing these vertical lines.
    # The toll at x=i depends on floor(y) at that point.
    # The path spans y-coordinates between sy+0.5 and ty+0.5.
    # It must therefore visit y-levels j+0.5 where j is an integer between min(sy, ty) and max(sy, ty).
    # We assume the path can be constructed to cross x=i at a y-level j+0.5 where j is
    # available in the range [min(sy, ty), max(sy, ty)], chosen to minimize toll.

    # If sy and ty have different parity, the range [min(sy, ty), max(sy, ty)] contains integers of both parities.
    # For any vertical line x=i to be crossed, we can choose a y-level j+0.5 within the vertical span such that j has the desired parity.
    # Desired parity for j is (i % 2) ^ 1 (different from i % 2) to make i + j odd (toll 0).
    # Since both parities of j are available, we can always choose j such that i+j is odd.
    # Minimum horizontal toll is 0 in this case.
    if (sy % 2) != (ty % 2):
        horizontal_toll = 0
    else:
        # sy and ty have the same parity.
        # The y-levels j+0.5 visited will have j with the same parity as sy.
        # Toll at x=i is paid if i + j is even. Since j has parity sy % 2, this means toll if i % 2 == sy % 2.
        # We need to count integers i in [min(sx, tx) + 1, max(sx, tx)] that have the same parity as sy.
        x_start_cross = min(sx, tx) + 1
        x_end_cross = max(sx, tx)

        if x_start_cross > x_end_cross:
            horizontal_toll = 0
        else:
            total_numbers_crossed = x_end_cross - x_start_cross + 1
            desired_parity = sy % 2

            # Count integers i in [x_start_cross, x_end_cross] where i % 2 == desired_parity
            first_i = x_start_cross
            first_i_parity = first_i % 2

            if first_i_parity == desired_parity:
                # The first number has the desired parity. The numbers with desired parity are first, third, fifth, etc.
                # Number of such integers is ceil(total_numbers_crossed / 2)
                horizontal_toll = (total_numbers_crossed + 1) // 2
            else:
                # The first number does not have the desired parity. The numbers with desired parity are second, fourth, sixth, etc.
                # Number of such integers is floor(total_numbers_crossed / 2)
                horizontal_toll = total_numbers_crossed // 2

    total_toll = vertical_toll + horizontal_toll
    print(total_toll)

# The code should directly read from stdin and print to stdout.
# The solve() function contains the core logic.
# The standard input/output setup is handled by sys.stdin.readline() and print().

solve()