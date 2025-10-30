def main():
    # Create a mapping dictionary for each direction and its opposite
    opposite = {
        "N": "S",
        "S": "N",
        "E": "W",
        "W": "E",
        "NE": "SW",
        "SW": "NE",
        "NW": "SE",
        "SE": "NW"
    }
    
    # Read the input: direction string
    D = input().strip()
    
    # Output the corresponding opposite direction
    print(opposite[D])

# Call the main function to execute our solution
main()