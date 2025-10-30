import sys

# Read the input string
D = sys.stdin.readline().strip()

# Create a dictionary mapping each direction to its opposite
opposite_directions = {
    'N': 'S',
    'S': 'N',
    'E': 'W',
    'W': 'E',
    'NE': 'SW',
    'SW': 'NE',
    'NW': 'SE',
    'SE': 'NW',
}

# Find the opposite direction
opposite_D = opposite_directions[D]

# Print the result
print(opposite_D)