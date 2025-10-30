# Read the input direction
d = input().strip()

# Define the mapping from each direction to its opposite
opposites = {
    'N': 'S',
    'S': 'N',
    'E': 'W',
    'W': 'E',
    'NE': 'SW',
    'SW': 'NE',
    'NW': 'SE',
    'SE': 'NW'
}

# Output the corresponding opposite direction
print(opposites[d])