def main():
    import sys
    # Read the input direction from stdin, stripping any whitespace/newlines.
    D = sys.stdin.readline().strip()
    
    # Dictionary mapping each direction to its opposite.
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
    
    # Output the opposite direction.
    print(opposites[D])

# Invoke the main function.
main()