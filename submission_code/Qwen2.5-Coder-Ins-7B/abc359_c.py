# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

Sx = int(data[0])
Sy = int(data[1])
Tx = int(data[2])
Ty = int(data[3])

# Function to determine the tile type
def tile_type(x, y):
    return (x + y) % 2

# Function to calculate the minimum toll
def min_toll(Sx, Sy, Tx, Ty):
    # Calculate the number of moves needed to reach the target tile
    moves = abs(Tx - Sx) + abs(Ty - Sy)
    # If the starting and ending tiles are the same, no toll is needed
    if tile_type(Sx, Sy) == tile_type(Tx, Ty):
        return 0
    # Otherwise, the minimum toll is the number of moves
    return moves

# Output the result
print(min_toll(Sx, Sy, Tx, Ty))