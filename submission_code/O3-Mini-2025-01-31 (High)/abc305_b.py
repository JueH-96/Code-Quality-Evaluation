def main():
    # Define the positions of the points along the line
    positions = {
        'A': 0,
        'B': 3,
        'C': 4,
        'D': 8,
        'E': 9,
        'F': 14,
        'G': 23
    }
    
    # Read input from standard input
    import sys
    input_line = sys.stdin.read().strip()
    if not input_line:
        return
    p, q = input_line.split()
    
    # Calculate the distance as the absolute difference between positions
    distance = abs(positions[p] - positions[q])
    
    # Print the result to standard output
    print(distance)

if __name__ == '__main__':
    main()