def main():
    import sys
    D = sys.stdin.readline().strip()
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
    # Print the direction opposite to D
    print(opposites[D])

if __name__ == "__main__":
    main()