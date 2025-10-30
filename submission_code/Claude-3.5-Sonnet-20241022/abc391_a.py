# Read input direction
D = input().strip()

# Dictionary mapping directions to their opposites
opposite_directions = {
    'N': 'S',
    'S': 'N',
    'E': 'W',
    'W': 'E',
    'NE': 'SW',
    'NW': 'SE',
    'SE': 'NW',
    'SW': 'NE'
}

# Print the opposite direction
print(opposite_directions[D])