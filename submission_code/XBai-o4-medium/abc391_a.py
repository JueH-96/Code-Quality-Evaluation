# Read the input direction
d = input().strip()

# Define the opposite directions
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

# Output the opposite direction
print(opposite[d])