# YOUR CODE HERE
def solve(V1, V2, V3):
    # Total volume of one cube
    cube_volume = 7**3
    
    # Total volume covered by at least one cube
    total_volume = V1 + V2 + V3
    
    # Check if the total volume is valid
    if total_volume != 3 * cube_volume - 3 * V2 - 2 * V3:
        return "No"
    
    # Try to find a configuration
    # We will place the cubes in a way that minimizes overlap
    # Start with the first cube at (0, 0, 0)
    a1, b1, c1 = 0, 0, 0
    
    # Place the second cube such that it overlaps with the first cube
    # in a way that creates the required V2 and V3
    a2, b2, c2 = 0, 7 - V3, 0
    
    # Place the third cube such that it overlaps with the first and second cubes
    # in a way that creates the required V2 and V3
    a3, b3, c3 = 7 - V3, 0, 0
    
    # Check the volumes
    # Volume of the region contained in exactly one of the cubes
    V1_calculated = 3 * cube_volume - 3 * V2 - 2 * V3
    
    # Volume of the region contained in exactly two of the cubes
    V2_calculated = 3 * V3 + V2
    
    # Volume of the region contained in all three cubes
    V3_calculated = V3
    
    if V1_calculated == V1 and V2_calculated == V2 and V3_calculated == V3:
        return f"Yes
{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}"
    
    return "No"

import sys
input = sys.stdin.read().strip().split()
V1 = int(input[0])
V2 = int(input[1])
V3 = int(input[2])

print(solve(V1, V2, V3))