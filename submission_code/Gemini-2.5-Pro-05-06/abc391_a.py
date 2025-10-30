# Read the input direction string
D = input()

# Define the mapping of directions to their opposites
opposites = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
    "NE": "SW",
    "SW": "NE",
    "NW": "SE",
    "SE": "NW"
}

# Find the opposite direction from the dictionary
opposite_direction = opposites[D]

# Print the result
print(opposite_direction)