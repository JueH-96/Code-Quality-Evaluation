# YOUR CODE HERE
import sys

def solve():
    # Read the input direction from standard input
    d = sys.stdin.readline().strip()

    # Define a dictionary mapping each direction to its opposite
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

    # Look up the opposite direction in the dictionary
    opposite_d = opposites[d]

    # Print the opposite direction to standard output
    print(opposite_d)

# Call the solve function to execute the logic
solve()
# YOUR CODE HERE