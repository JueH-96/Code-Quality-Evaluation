def main():
    import sys
    # Read the input, we expect 4 integers input on one line
    data = sys.stdin.read().strip().split()
    # Convert input tokens to integers
    balls = list(map(int, data))
    
    # Create a frequency dictionary for occurrences of each ball color
    frequency = {}
    for color in balls:
        frequency[color] = frequency.get(color, 0) + 1
    
    # Calculate the maximum number of operations
    result = 0
    for count in frequency.values():
        result += count // 2
    
    # Print the result
    print(result)

if __name__ == '__main__':
    main()