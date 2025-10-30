def max_operations(balls):
    from collections import Counter
    
    # Count the occurrences of each color
    color_count = Counter(balls)
    
    # Calculate the maximum number of operations
    operations = 0
    for count in color_count.values():
        operations += count // 2  # Each pair can be discarded
    
    return operations

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    # Read the input
    balls = list(map(int, input().strip().split()))
    # Get the result
    result = max_operations(balls)
    # Print the result
    print(result)