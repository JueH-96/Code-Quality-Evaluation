# YOUR CODE HERE
import sys

def min_toll(sx, sy, tx, ty):
    # Calculate the difference in x and y coordinates
    dx = abs(tx - sx)
    dy = abs(ty - sy)
    
    # Calculate the minimum toll
    # The toll is the sum of the differences in x and y coordinates divided by 2
    # because each tile covers 2 units in one direction.
    toll = (dx + dy) // 2
    
    # If the sum of the coordinates is odd, we need to add 1 to the toll
    if (sx + sy + tx + ty) % 2 == 1:
        toll += 1
    
    return toll

# Read input from stdin
sx, sy = map(int, input().split())
tx, ty = map(int, input().split())

# Calculate and print the minimum toll
print(min_toll(sx, sy, tx, ty))