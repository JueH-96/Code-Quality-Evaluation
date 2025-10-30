def main():
    import sys
    from collections import Counter
    
    # Read all input data from stdin, strip and split into tokens
    data = sys.stdin.read().strip().split()
    
    # Convert tokens into integers (colors of balls)
    balls = list(map(int, data))
    
    # Count the occurrences of each color
    counts = Counter(balls)
    
    # For each color, the maximum pairs we can remove is count // 2.
    # Summing up these counts gives the maximum number of operations.
    operations = sum(count // 2 for count in counts.values())
    
    # Print the total number of operations possible.
    print(operations)

# Call the main function.
if __name__ == "__main__":
    main()