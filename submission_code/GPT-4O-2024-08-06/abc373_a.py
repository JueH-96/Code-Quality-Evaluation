# YOUR CODE HERE
import sys

def count_matching_lengths():
    # Read all input lines
    lines = sys.stdin.read().strip().split('
')
    
    # Initialize a counter for the number of matches
    count = 0
    
    # Iterate over each string with its index
    for i, s in enumerate(lines, start=1):
        # Check if the length of the string matches its index
        if len(s) == i:
            count += 1
    
    # Print the result
    print(count)

# Call the function to execute the logic
count_matching_lengths()