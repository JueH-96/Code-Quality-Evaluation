# YOUR CODE HERE
def opposite_direction(D):
    # Define a dictionary mapping each direction to its opposite
    opposite = {
        'N': 'S',
        'S': 'N',
        'E': 'W',
        'W': 'E',
        'NE': 'SW',
        'SW': 'NE',
        'NW': 'SE',
        'SE': 'NW'
    }
    
    # Return the opposite direction
    return opposite[D]

# Read input from standard input
import sys
input = sys.stdin.read().strip()

# Get the opposite direction
result = opposite_direction(input)

# Print the result
print(result)