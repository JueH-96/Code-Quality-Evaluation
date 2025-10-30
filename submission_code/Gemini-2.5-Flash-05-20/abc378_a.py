import sys
from collections import Counter

def solve():
    # Read the input line. It contains four space-separated integers.
    # sys.stdin.readline() is generally faster than input() for competitive programming,
    # though for a single line of 4 integers, the difference is negligible.
    line = sys.stdin.readline().strip()
    
    # Split the string by spaces and convert each part to an integer.
    # The result is a list of integers representing the colors of the four balls.
    colors = list(map(int, line.split()))
    
    # Use collections.Counter to efficiently count the occurrences of each color.
    # For example, if colors is [2, 1, 2, 1], color_counts will be Counter({2: 2, 1: 2}).
    color_counts = Counter(colors)
    
    # Initialize a variable to store the total number of operations performed.
    total_operations = 0
    
    # Iterate through the counts of each unique color.
    # .values() gives an iterable of just the counts (e.g., 2, 2 from the example above).
    for count in color_counts.values():
        # For each color, the number of pairs we can form is its count divided by 2
        # using integer division (//). This discards any single leftover ball.
        total_operations += count // 2
        
    # Print the final result, which is the maximum number of operations.
    print(total_operations)

# Call the solve function to execute the program logic.
solve()