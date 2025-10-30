def main():
    import sys

    # Read the direction from input and strip any whitespace/newlines
    D = sys.stdin.readline().strip()

    # Mapping of each direction to its opposite
    opposite = {
        "N":  "S",
        "S":  "N",
        "E":  "W",
        "W":  "E",
        "NE": "SW",
        "SW": "NE",
        "NW": "SE",
        "SE": "NW"
    }

    # Print the opposite direction
    print(opposite[D])


if __name__ == "__main__":
    main()