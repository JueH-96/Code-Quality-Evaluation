# Read inputs from stdin
s_str = input()
t_str = input()

# Mapping from vertex character to an integer (0-4).
# This assumes a consecutive labeling of vertices around the pentagon,
# e.g., A, B, C, D, E in order (clockwise or counter-clockwise).
mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

# Function to determine the "characteristic" of a line segment in the pentagon.
# This characteristic value distinguishes between sides and diagonals.
# It represents the minimum number of edges one must traverse along the
# perimeter to get from one vertex to the other.
# For a side (e.g., A-B), this value is 1.
# For a diagonal (e.g., A-C), this value is 2.
def get_segment_characteristic(p1_char, p2_char, vertex_map):
    v1 = vertex_map[p1_char]
    v2 = vertex_map[p2_char]
    
    # Calculate the absolute difference in their numerical labels
    diff = abs(v1 - v2)
    
    # The characteristic is the shorter path along the perimeter:
    # either 'diff' steps or '5 - diff' steps (since there are 5 vertices in a pentagon).
    characteristic = min(diff, 5 - diff)
    return characteristic

# Extract the two characters representing the points for segment S
s1_char = s_str[0]
s2_char = s_str[1]

# Extract the two characters representing the points for segment T
t1_char = t_str[0]
t2_char = t_str[1]

# Calculate the characteristic for segment S
s_characteristic = get_segment_characteristic(s1_char, s2_char, mapping)
# Calculate the characteristic for segment T
t_characteristic = get_segment_characteristic(t1_char, t2_char, mapping)

# Two segments in a regular pentagon have equal length if and only if
# they are of the same type (both sides or both diagonals).
# The 'characteristic' value (1 for side, 2 for diagonal) captures this type.
if s_characteristic == t_characteristic:
    print("Yes")
else:
    print("No")