import sys

def solve():
    # Read the input direction
    D = sys.stdin.readline().strip()

    # Define the mapping of directions to their opposites
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

    # Look up the opposite direction
    opposite_D = opposite_directions[D]

    # Print the result
    print(opposite_D)

if __name__ == "__main__":
    solve()