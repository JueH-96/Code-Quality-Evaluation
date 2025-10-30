# YOUR CODE HERE
import sys

def solve():
    """
    Reads three age comparison operators and determines the middle brother.
    """
    # Read the three space-separated comparison strings from a single line.
    try:
        s_ab, s_ac, s_bc = sys.stdin.readline().strip().split()
    except (IOError, ValueError):
        # Handle potential empty input or incorrect format, though
        # problem constraints suggest this won't happen.
        return

    # Check if A is the middle brother.
    # This occurs if A is older than one brother and younger than the other.
    # The two possible orderings are B < A < C and C < A < B.
    # B < A < C translates to S_AB='>' and S_AC='<'.
    # C < A < B translates to S_AB='<' and S_AC='>'.
    if (s_ab == '>' and s_ac == '<') or \
       (s_ab == '<' and s_ac == '>'):
        print('A')

    # Check if B is the middle brother.
    # The two possible orderings are A < B < C and C < B < A.
    # A < B < C translates to S_AB='<' and S_BC='<'.
    # C < B < A translates to S_AB='>' and S_BC='>'.
    elif (s_ab == '<' and s_bc == '<') or \
         (s_ab == '>' and s_bc == '>'):
        print('B')

    # If neither A nor B is the middle brother, C must be.
    # The problem constraints guarantee that a unique solution exists,
    # so an else case is sufficient to cover the remaining possibilities for C.
    # For completeness, the orderings would be A < C < B or B < C < A.
    else:
        print('C')

solve()