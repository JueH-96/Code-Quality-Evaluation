# YOUR CODE HERE
import sys

# Define the sets of normalized sides and diagonals for a regular pentagon.
# Vertices are A, B, C, D, E in clockwise order.
# Normalization means sorting the two endpoints alphabetically.
# Sides connect adjacent vertices along the perimeter: AB, BC, CD, DE, EA.
# Normalized sides (sorted alphabetically): ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('A', 'E').
sides_normalized = {('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('A', 'E')}

# Diagonals connect non-adjacent vertices: AC, AD, BD, BE, CE.
# Normalized diagonals (sorted alphabetically): ('A', 'C'), ('A', 'D'), ('B', 'D'), ('B', 'E'), ('C', 'E').
diagonals_normalized = {('A', 'C'), ('A', 'D'), ('B', 'D'), ('B', 'E'), ('C', 'E')}

# Function to determine the type of segment (side or diagonal)
def get_segment_type(p1_char, p2_char):
    """
    Determines if the segment connecting p1_char and p2_char in a regular
    pentagon is a side or a diagonal, assuming vertices are A, B, C, D, E
    sequentially around the perimeter.

    Args:
        p1_char: The first endpoint character (A-E).
        p2_char: The second endpoint character (A-E).

    Returns:
        'side' if it's a side, 'diagonal' if it's a diagonal.
        Assumes p1_char != p2_char and p1_char, p2_char are in 'A' through 'E'.
    """
    # Normalize the pair of characters by sorting them alphabetically
    # This makes 'AB' equivalent to 'BA', 'AC' to 'CA', etc.
    p1, p2 = sorted([p1_char, p2_char])
    pair = (p1, p2)

    # Check if the normalized pair is in the set of sides or diagonals
    if pair in sides_normalized:
        return 'side'
    elif pair in diagonals_normalized:
        return 'diagonal'
    else:
        # This branch should not be reached with valid input according to constraints
        # (distinct vertices from A-E). Any pair of distinct vertices in a pentagon
        # must form either a side or a diagonal.
        # Returning None or raising an error would also be valid handling here.
        # For this problem, we can assume valid input pairs will always be sides or diagonals.
        return 'unknown' # Indicates an unexpected input pair

# Read the two segments from standard input
s1s2_str = sys.stdin.readline().strip()
t1t2_str = sys.stdin.readline().strip()

# Extract the endpoints of the two segments
s1, s2 = s1s2_str[0], s1s2_str[1]
t1, t2 = t1t2_str[0], t1t2_str[1]

# Determine the type (side or diagonal) of each segment
type_s = get_segment_type(s1, s2)
type_t = get_segment_type(t1, t2)

# In a regular pentagon, all sides have the same length, and all diagonals
# have the same (different) length.
# Therefore, two segments have equal length if and only if they are of the same type
# (both sides or both diagonals).
if type_s == type_t:
    print("Yes")
else:
    print("No")