import math

def solve():
    R_in = int(input())

    # Using floating point numbers for geometric calculations
    R_float = float(R_in)
    R_sq_float = R_float * R_float # R^2

    total_squares = 0 # This must be an integer count

    # Iterate X_int from 0 up to R_in-1.
    # X_int represents |i| from the problem statement, where square center is (i,j).
    # The condition for a square to be fully contained is that its corner furthest
    # from the origin, which is at distance sqrt((|i|+0.5)^2 + (|j|+0.5)^2), must be <= R.
    # So, (X_int+0.5)^2 + (Y_int+0.5)^2 <= R^2.
    #
    # The loop for X_int goes from 0 up to floor(R - 0.5).
    # Since R_in is an integer, floor(R_in - 0.5) is R_in - 1.
    # So, X_int iterates from 0 to R_in - 1, which is range(R_in).
    
    for X_int in range(R_in):
        # Current value for |i| is X_int.
        # x_coord_of_corner refers to (|i|+0.5)
        x_coord_of_corner = float(X_int) + 0.5
        x_coord_of_corner_sq = x_coord_of_corner * x_coord_of_corner

        # Now we need to find Y_int (which is |j|) such that
        # (Y_int+0.5)^2 <= R_sq_float - x_coord_of_corner_sq
        remaining_sq_budget_for_Y = R_sq_float - x_coord_of_corner_sq

        # Smallest possible (Y_int+0.5)^2 for Y_int >= 0 is (0+0.5)^2 = 0.25.
        # If remaining_sq_budget_for_Y < 0.25, no non-negative integer Y_int can satisfy the condition.
        if remaining_sq_budget_for_Y < 0.25:
            # This implies that even for Y_int=0, the condition is not met.
            # So, no Y_int >= 0 works for this X_int.
            # Y_max_for_this_X will be <0 (e.g. -1), handled by the conditions below.
            Y_max_for_this_X = -1 
        else:
            # (Y_int+0.5) <= sqrt(remaining_sq_budget_for_Y)
            # Y_int <= sqrt(remaining_sq_budget_for_Y) - 0.5
            # Since Y_int must be an integer, Y_max = floor(sqrt(remaining_sq_budget_for_Y) - 0.5)
            # Ensure result is int for summation. math.floor returns float.
            Y_max_for_this_X = int(math.floor(math.sqrt(remaining_sq_budget_for_Y) - 0.5))

        # If Y_max_for_this_X < 0, it means no Y_int >= 0 works for this X_int. Contribution is 0.
        # Otherwise, Y_int can range from 0 to Y_max_for_this_X.
        if Y_max_for_this_X >= 0:
            # Case 1: X_int = 0 (square's center x-coordinate is 0)
            if X_int == 0:
                # If Y_int = 0: one square (0,0).
                # If Y_int > 0: two squares (0, Y_int) and (0, -Y_int) for each Y_int from 1 to Y_max_for_this_X.
                # Number of such Y_int > 0 values is Y_max_for_this_X.
                # Total contribution for X_int=0: 1 (for Y_int=0) + 2 * Y_max_for_this_X.
                total_squares += (1 + 2 * Y_max_for_this_X)
            # Case 2: X_int > 0 (square's center x-coordinate is non-zero)
            else: # X_int > 0
                # If Y_int = 0: two squares (X_int, 0) and (-X_int, 0).
                # If Y_int > 0: four squares (+/-X_int, +/-Y_int) for each Y_int from 1 to Y_max_for_this_X.
                # Number of such Y_int > 0 values is Y_max_for_this_X.
                # Total contribution for X_int > 0: 2 (for Y_int=0) + 4 * Y_max_for_this_X.
                total_squares += (2 + 4 * Y_max_for_this_X)
        # Optimization: if Y_max_for_this_X < 0 for current X_int > 0, then all subsequent (larger) X_int
        # will also have Y_max < 0 (or remaining_sq_budget_for_Y < 0.25). So we can break.
        # (For X_int=0, Y_max_for_this_X is always >=0 if R_in >= 1.)
        elif X_int > 0: # This 'elif' implies Y_max_for_this_X < 0 due to the 'if' above
            break

    print(total_squares)

solve()