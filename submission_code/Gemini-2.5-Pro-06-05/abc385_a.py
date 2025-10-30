# YOUR CODE HERE
def solve():
    """
    Reads three integers A, B, C and determines if they can be divided
    into two or more groups with equal sums.
    """
    try:
        # Read the three integers from a single line of standard input.
        A, B, C = map(int, input().split())

        # Check for partition into two groups with equal sums.
        # This is possible if one number is the sum of the other two.
        # e.g., groups {A} and {B, C} have equal sums if A = B + C.
        is_two_group_possible = (A + B == C) or (A + C == B) or (B + C == A)

        # Check for partition into three groups with equal sums.
        # This is possible if all three numbers are equal.
        # The groups would be {A}, {B}, {C}.
        is_three_group_possible = (A == B and B == C)

        # If either a 2-group or 3-group partition is possible, the answer is "Yes".
        if is_two_group_possible or is_three_group_possible:
            print("Yes")
        else:
            print("No")
            
    except (IOError, ValueError):
        # This handles potential errors with input, which is good practice
        # but not strictly necessary for most online judge platforms.
        pass

# Run the solution
solve()