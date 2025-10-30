# YOUR CODE HERE
D = input().strip()

opposite_directions = {
    "N": "S",
    "S": "N",
    "E": "W",
    "W": "E",
    "NE": "SW",
    "SW": "NE",
    "NW": "SE",
    "SE": "NW"
}

print(opposite_directions[D])