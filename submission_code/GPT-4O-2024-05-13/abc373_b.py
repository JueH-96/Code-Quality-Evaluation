# YOUR CODE HERE
def minimal_traveled_distance(S):
    # Create a dictionary to map each character to its position
    position = {char: idx for idx, char in enumerate(S)}
    
    # Start at the position of 'A'
    current_pos = position['A']
    total_distance = 0
    
    # Iterate through the characters from 'A' to 'Z'
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        next_pos = position[char]
        total_distance += abs(next_pos - current_pos)
        current_pos = next_pos
    
    return total_distance

# Read input
import sys
input = sys.stdin.read().strip()

# Calculate and print the result
print(minimal_traveled_distance(input))