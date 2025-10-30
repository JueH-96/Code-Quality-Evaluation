import sys

def solve():
    # Read the two input lines
    s_segment_str = sys.stdin.readline().strip()
    t_segment_str = sys.stdin.readline().strip()

    # Map characters (vertices) to numerical indices
    # This allows us to easily calculate the 'distance' between vertices
    # along the perimeter of the pentagon.
    char_to_int = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    # A regular pentagon has 5 vertices
    num_vertices = 5

    # Helper function to determine the type/length of a segment
    # In a regular pentagon, there are only two distinct lengths for segments
    # connecting two distinct vertices:
    # 1. Side length (connecting adjacent vertices, e.g., AB, BC)
    # 2. Diagonal length (connecting non-adjacent vertices, e.g., AC, AD)
    # This function returns 1 for a side and 2 for a diagonal.
    def get_segment_length_type(p1_char, p2_char):
        # Get numerical indices for the two points
        p1_idx = char_to_int[p1_char]
        p2_idx = char_to_int[p2_char]

        # Calculate the absolute difference between their indices
        diff = abs(p1_idx - p2_idx)
        
        # The 'length type' is the minimum number of steps along the perimeter
        # to go from p1 to p2 (either clockwise or counter-clockwise).
        # For example, A(0) to E(4): diff=4. min(4, 5-4) = min(4,1) = 1. (Side)
        # A(0) to C(2): diff=2. min(2, 5-2) = min(2,3) = 2. (Diagonal)
        segment_type = min(diff, num_vertices - diff)
        return segment_type

    # Extract the two characters for the first segment (S1S2)
    s1_char = s_segment_str[0]
    s2_char = s_segment_str[1]
    
    # Extract the two characters for the second segment (T1T2)
    t1_char = t_segment_str[0]
    t2_char = t_segment_str[1]

    # Determine the length type for both segments
    s_segment_type = get_segment_length_type(s1_char, s2_char)
    t_segment_type = get_segment_length_type(t1_char, t2_char)

    # If the length types are the same, their actual lengths are the same.
    if s_segment_type == t_segment_type:
        print("Yes")
    else:
        print("No")

# Call the solve function to run the program
solve()