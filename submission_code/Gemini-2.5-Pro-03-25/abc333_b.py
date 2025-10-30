# YOUR CODE HERE
import sys

def calculate_distance(v1_char, v2_char):
    """
    Calculates the shortest distance between two vertices along the perimeter 
    of a regular pentagon with vertices labeled 'A' through 'E' in order.
    The distance is defined as the minimum number of edges one must traverse 
    along the perimeter to get from one vertex to the other.
    
    In a regular pentagon, there are only two distinct lengths for segments 
    connecting vertices:
    1. Side length: This corresponds to a distance of 1 (adjacent vertices).
    2. Diagonal length: This corresponds to a distance of 2 (vertices separated 
       by one vertex along the perimeter).
       
    Args:
        v1_char (str): The character representing the first vertex (e.g., 'A').
        v2_char (str): The character representing the second vertex (e.g., 'C').

    Returns:
        int: The shortest distance (1 or 2) between the vertices along the perimeter.
    """
    # Get the ASCII values of the characters. Since 'A', 'B', 'C', 'D', 'E' are
    # consecutive characters, their ASCII values are also consecutive integers.
    # This allows us to treat them as numerical indices 0-4 implicitly.
    # ord('A')=65, ord('B')=66, ..., ord('E')=69.
    o1 = ord(v1_char)
    o2 = ord(v2_char)
    
    # Calculate the absolute difference in their ASCII values.
    # This represents the distance traveling in one direction along the perimeter.
    # For example, diff('A', 'C') = ord('C') - ord('A') = 67 - 65 = 2.
    # For example, diff('A', 'E') = ord('E') - ord('A') = 69 - 65 = 4.
    diff = abs(o1 - o2)
    
    # The shortest distance along the perimeter is the minimum of the distance
    # in one direction (diff) and the distance in the other direction (5 - diff).
    # The total number of vertices is 5.
    # Example: Distance between A and E. diff = 4. The other path length is 5-4 = 1.
    # The minimum is min(4, 1) = 1. This corresponds to the edge AE.
    # Example: Distance between A and D. diff = 3. The other path length is 5-3 = 2.
    # The minimum is min(3, 2) = 2. This corresponds to the diagonal AD.
    distance = min(diff, 5 - diff)
    
    # The distance will be 1 for adjacent vertices (side length) and 
    # 2 for non-adjacent vertices (diagonal length).
    return distance

def solve():
    """
    Reads the two segment endpoint pairs from standard input, 
    calculates the distance for each segment based on its endpoints 
    on the regular pentagon, and determines if the lengths are equal.
    Prints "Yes" if the lengths (distances) are equal, "No" otherwise.
    """
    # Read the first input line, which contains two characters representing the endpoints S1 and S2.
    s_str = sys.stdin.readline().strip()
    # Read the second input line, which contains two characters representing the endpoints T1 and T2.
    t_str = sys.stdin.readline().strip()

    # Extract the vertex characters for the first segment S1S2
    s1 = s_str[0]
    s2 = s_str[1]
    
    # Extract the vertex characters for the second segment T1T2
    t1 = t_str[0]
    t2 = t_str[1]

    # Calculate the distance corresponding to the first segment S1S2
    dist_s = calculate_distance(s1, s2)
    
    # Calculate the distance corresponding to the second segment T1T2
    dist_t = calculate_distance(t1, t2)

    # In a regular pentagon, two segments connecting vertices have equal length 
    # if and only if the shortest distance between their respective endpoints 
    # along the perimeter is equal.
    if dist_s == dist_t:
        # If the distances are equal, the segment lengths are equal.
        print("Yes")
    else:
        # If the distances are different, the segment lengths are different.
        print("No")

# Execute the main solution function when the script is run
solve()