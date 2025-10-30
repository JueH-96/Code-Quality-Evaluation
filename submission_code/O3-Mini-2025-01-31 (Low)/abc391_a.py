def main():
    import sys
    input_str = sys.stdin.read().strip()
    
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
    
    # Assuming input_str is exactly one of the keys in opposites
    if input_str in opposites:
        print(opposites[input_str])
    else:
        # In case of extra whitespace or multiple inputs, splitting
        # but problem statement guarantees valid single string input.
        parts = input_str.split()
        if parts and parts[0] in opposites:
            print(opposites[parts[0]])

if __name__ == "__main__":
    main()