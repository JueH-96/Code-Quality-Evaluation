# Read the input from stdin
direction = input()

# Define the opposite directions
opposite_directions = {
    "N": "S",
    "E": "W",
    "W": "E",
    "S": "N",
    "NE": "SW",
    "NW": "SE",
    "SE": "NW",
    "SW": "NE"
}

# Print the opposite direction
print(opposite_directions[direction])