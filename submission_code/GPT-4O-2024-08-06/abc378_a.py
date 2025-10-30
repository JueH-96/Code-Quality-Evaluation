# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = list(map(int, input().strip().split()))
    
    # Dictionary to count occurrences of each color
    color_count = {}
    
    # Count each color
    for color in data:
        if color in color_count:
            color_count[color] += 1
        else:
            color_count[color] = 1
    
    # Calculate the maximum number of operations
    max_operations = 0
    for count in color_count.values():
        max_operations += count // 2
    
    # Print the result
    print(max_operations)