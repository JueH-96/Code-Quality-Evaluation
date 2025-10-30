def find_cubes(V1, V2, V3):
    # The side length of the cubes
    side_length = 7
    # The volume of a single cube
    cube_volume = side_length ** 3
    
    # Check if the volumes are valid
    if V1 + V2 + V3 > 3 * cube_volume:
        return "No"
    
    # We will try to find a configuration that satisfies the conditions
    # We can use a simple configuration for testing
    # Let's place the cubes in a way that they overlap correctly
    # We will use a configuration that is easy to calculate
    
    # Example configuration:
    # C1 at (0, 0, 0)
    # C2 at (0, 0, 7)
    # C3 at (0, 7, 0)
    
    # This gives us:
    # V3 = volume of intersection of all three cubes
    # V2 = volume of intersection of any two cubes
    # V1 = volume of the regions contained in exactly one cube
    
    # Calculate the volumes based on this configuration
    # The intersection of all three cubes is a line segment of length 0
    # The intersection of any two cubes is a square of area 7*7
    # The volume of one cube is 7*7*7 = 343
    
    # Let's calculate the volumes
    a1, b1, c1 = 0, 0, 0
    a2, b2, c2 = 0, 0, 7
    a3, b3, c3 = 0, 7, 0
    
    # Calculate the volumes
    V3_calculated = 0  # No volume from the intersection of all three
    V2_calculated = 2 * (side_length * side_length * (side_length - 1))  # Two pairs of cubes overlap
    V1_calculated = 3 * cube_volume - V2_calculated - V3_calculated  # Remaining volume in single cubes
    
    # Check if the calculated volumes match the input volumes
    if V1_calculated == V1 and V2_calculated == V2 and V3_calculated == V3:
        return f"Yes
{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}"
    
    return "No"

import sys
input = sys.stdin.read
V1, V2, V3 = map(int, input().strip().split())
result = find_cubes(V1, V2, V3)
print(result)