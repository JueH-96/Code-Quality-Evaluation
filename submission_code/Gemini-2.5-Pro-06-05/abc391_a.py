# YOUR CODE HERE
def solve():
    """
    Reads a direction string and prints its opposite.
    """
    # Read the input direction from stdin
    D = input()

    # A dictionary to store the mapping of a direction to its opposite
    opposite_map = {
        "N": "S",
        "S": "N",
        "E": "W",
        "W": "E",
        "NE": "SW",
        "NW": "SE",
        "SE": "NW",
        "SW": "NE"
    }

    # Look up the opposite direction in the dictionary and print it to stdout
    print(opposite_map[D])

# Call the solve function to run the program
solve()