# YOUR CODE HERE
import sys

# Read input from stdin
S = sys.stdin.read().strip()

# Initialize the starting position of the finger
start_position = 0  # A is at position 0

# Initialize the total traveled distance
total_distance = 0

# Iterate through the characters in the string S
for char in S:
    # Calculate the position of the current character
    current_position = ord(char) - ord('A')
    
    # Calculate the distance traveled to reach the current character
    distance = abs(current_position - start_position)
    
    # Add the distance to the total traveled distance
    total_distance += distance
    
    # Update the starting position to the current character's position
    start_position = current_position

# Print the total traveled distance
print(total_distance)