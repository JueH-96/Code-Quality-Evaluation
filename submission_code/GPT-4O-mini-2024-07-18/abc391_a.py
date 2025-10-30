# YOUR CODE HERE
def opposite_direction(direction):
    opposites = {
        'N': 'S',
        'E': 'W',
        'W': 'E',
        'S': 'N',
        'NE': 'SW',
        'NW': 'SE',
        'SE': 'NW',
        'SW': 'NE'
    }
    return opposites[direction]

# Read input
D = input().strip()

# Get and print the opposite direction
print(opposite_direction(D))