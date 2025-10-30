# Reading the input direction from standard input
import sys
input = sys.stdin.read().strip()

# Dictionary to map each direction to its opposite
opposite_directions = {
    'N': 'S',
    'S': 'N',
    'E': 'W',
    'W': 'E',
    'NE': 'SW',
    'SW': 'NE',
    'NW': 'SE',
    'SE': 'NW'
}

# Output the opposite direction
print(opposite_directions[input])