# YOUR CODE HERE
# Read the input direction string
D = input()

# Create a dictionary to map each direction to its opposite
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

# Look up the opposite direction using the input string as the key
# The problem guarantees that D will be one of the keys in the dictionary
opposite = opposite_directions[D]

# Print the opposite direction
print(opposite)