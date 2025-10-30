# YOUR CODE HERE
import math
import sys

def solve():
    """
    Solves the problem of counting squares completely contained within a circle.

    Reads the radius R from stdin.
    The problem asks for the number of integer pairs (i, j) such that the square
    centered at (i, j) with corners at (i+/-0.5, j+/-0.5) is entirely contained
    within the circle of radius R centered at the origin (0, 0).

    A square is completely contained if all its points are within distance R
    from the origin. This is equivalent to checking the point farthest from the
    origin within the square. The farthest points are the corners.
    Specifically, the corner farthest from the origin is (|i|+0.5, |j|+0.5)
    (or symmetrically equivalent points based on signs of i, j).
    So, the condition is:
    distance((|i|+0.5, |j|+0.5), (0,0)) <= R
    sqrt( (|i|+0.5)^2 + (|j|+0.5)^2 ) <= R
    (|i|+0.5)^2 + (|j|+0.5)^2 <= R^2

    To avoid floating-point issues, we can manipulate the inequality.
    Let's multiply by 4:
    4 * (|i|+0.5)^2 + 4 * (|j|+0.5)^2 <= 4 * R^2
    (2 * (|i|+0.5))^2 + (2 * (|j|+0.5))^2 <= (2*R)^2
    (2|i| + 1)^2 + (2|j| + 1)^2 <= 4R^2

    Let X = 2|i| + 1 and Y = 2|j| + 1. Both X and Y must be positive odd integers.
    The condition becomes X^2 + Y^2 <= 4R^2.

    We need to count the number of integer pairs (i, j) satisfying this.
    We can iterate through possible values of i and, for each i, find the number
    of possible values for j.

    The algorithm uses this integer-based condition and leverages symmetry.
    It iterates through non-negative values of i, calculates the number of
    corresponding j values, and sums them up, accounting for symmetry.
    """
    try:
        # Read the radius R from standard input
        r_str = sys.stdin.readline()
        # Handle potential empty input line
        if not r_str:
            # If input is empty, perhaps exit or handle as error.
            # Based on problem constraints, input is always provided.
            return
        r = int(r_str)
    except ValueError:
        # Handle cases where input is not a valid integer.
        # According to problem constraints, R is always an integer.
        # If error handling is needed, print to stderr or raise exception.
        # For competitive programming context, assume valid input.
        # print("Invalid input: R must be an integer.", file=sys.stderr)
        return # Exit gracefully if input is invalid

    # Basic input validation based on constraints (Optional but good practice)
    # if not (1 <= r <= 10**6):
    #     print(f"Input R={r} out of range [1, 10^6].", file=sys.stderr)
    #     return

    # Calculate the limit: 4 * R^2, using integer arithmetic.
    # Python integers handle arbitrary size, so overflow is not an issue for R <= 10^6.
    limit = 4 * r * r

    # Initialize the total count of valid squares (pairs (i, j))
    total_count = 0

    # Determine the maximum possible absolute value for i.
    # From (2|i|+1)^2 <= 4R^2, we get 2|i|+1 <= sqrt(4R^2) = 2R.
    # So, 2|i| <= 2R - 1, which implies |i| <= R - 0.5.
    # Since i must be an integer, the maximum value for |i| is floor(R - 0.5).
    # As R is given as an integer, max |i| = R - 1.
    # This holds for R >= 1. If R=1, max_abs_i = 0.
    max_abs_i = r - 1

    # --- Case i = 0 ---
    # Calculate the terms for i = 0.
    # X = 2*abs(0) + 1 = 1
    x_squared_i0 = 1 * 1
    # Calculate the remaining limit for Y^2: C = 4R^2 - X^2
    c_int_i0 = limit - x_squared_i0

    # Check if any j is possible for i=0. We need Y^2 <= c_int_i0.
    # Since Y = 2|j|+1 >= 1, we need Y^2 >= 1.
    # Thus, we need c_int_i0 >= 1 for any solution to exist.
    if c_int_i0 >= 1:
        # Find the maximum possible integer value for Y = 2|j|+1.
        # Y^2 <= c_int_i0 implies Y <= floor(sqrt(c_int_i0)).
        # math.isqrt calculates the integer square root efficiently.
        max_y_i0 = math.isqrt(c_int_i0)

        # Count the number of positive odd integers Y such that 1 <= Y <= max_y_i0.
        # The positive odd integers are 1, 3, 5, ...
        # If max_y_i0 = k, the odd integers are 1, 3, ..., k (if k odd) or k-1 (if k even).
        # The count is (k+1)//2.
        num_positive_odd_y_i0 = (max_y_i0 + 1) // 2

        # If num_positive_odd_y_i0 > 0, there are valid Y values.
        # Each valid positive odd Y corresponds to a unique abs(j) = (Y-1)/2.
        # The possible values for abs(j) are 0, 1, ..., num_positive_odd_y_i0 - 1.
        # We need to count the number of integer j values corresponding to these abs(j).
        # If abs(j) = 0, then j=0 (1 value).
        # If abs(j) = k > 0, then j = +k or j = -k (2 values).
        # Total count of j values = 1 (for j=0) + 2 * (number of positive abs(j) values)
        # Number of positive abs(j) values = num_positive_odd_y_i0 - 1 (since abs(j)=0 is one value)
        # Total count = 1 + 2 * (num_positive_odd_y_i0 - 1) = 1 + 2*num_positive_odd_y_i0 - 2 = 2 * num_positive_odd_y_i0 - 1.
        # This formula works correctly only if num_positive_odd_y_i0 >= 1.
        if num_positive_odd_y_i0 > 0:
             count_j_for_i0 = 2 * num_positive_odd_y_i0 - 1
             total_count += count_j_for_i0

    # --- Cases i > 0 ---
    # Iterate i from 1 up to max_abs_i.
    # We leverage symmetry: the count for i and -i is the same.
    for i in range(1, max_abs_i + 1):
        # Calculate X = 2|i|+1 for positive i. |i| = i.
        x = 2 * i + 1
        x_squared = x * x
        # Calculate the remaining limit for Y^2: C = 4R^2 - X^2
        c_int = limit - x_squared

        # Check if any j is possible for this i. Need Y^2 <= c_int, and Y >= 1.
        # Requires c_int >= 1.
        if c_int >= 1:
            # Find the maximum possible integer value for Y = 2|j|+1.
            max_y = math.isqrt(c_int)

            # Count the number of positive odd integers Y such that 1 <= Y <= max_y.
            num_positive_odd_y = (max_y + 1) // 2

            # If there are possible values for Y (and thus for j)
            if num_positive_odd_y > 0:
                # Calculate the number of j values for this specific abs(i) value.
                count_j = 2 * num_positive_odd_y - 1
                # Add the count for both +i and -i due to symmetry.
                total_count += 2 * count_j
        else:
            # If c_int < 1 for the current value of i, it means no Y >= 1 satisfies Y^2 <= c_int.
            # Since X increases with i, X^2 increases, and c_int decreases.
            # If c_int < 1 for this i, it will be even smaller for all larger i.
            # Therefore, no more solutions will be found for larger i. We can break the loop early.
            break

    # Print the final total count to standard output
    print(total_count)

# Ensure the solve function is called when the script is executed directly
if __name__ == '__main__':
    solve()