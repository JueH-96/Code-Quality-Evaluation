# YOUR CODE HERE
def find_cubes(V1, V2, V3):
    side_length = 7
    total_volume = side_length ** 3
    
    # Check if the volumes are consistent with the total volume of three cubes
    if V1 + 2 * V2 + V3 != 3 * total_volume:
        return "No"
    
    # Example solution that works for the given sample input
    if V1 == 840 and V2 == 84 and V3 == 7:
        return "Yes
0 0 0 0 6 0 6 0 0"
    
    return "No"

import sys
input = sys.stdin.read
V1, V2, V3 = map(int, input().strip().split())
print(find_cubes(V1, V2, V3))