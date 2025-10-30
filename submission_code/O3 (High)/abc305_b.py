def main():
    import sys
    
    # Read the two letters
    data = sys.stdin.read().strip().split()
    if len(data) != 2:
        return  # invalid input, but per problem statement this won't happen
    p, q = data
    
    # Pre-computed coordinates of each point along the line
    coords = {
        'A': 0,
        'B': 3,
        'C': 4,
        'D': 8,
        'E': 9,
        'F': 14,
        'G': 23
    }
    
    # Calculate and output the absolute distance
    distance = abs(coords[p] - coords[q])
    print(distance)

# Call the main function
if __name__ == "__main__":
    main()