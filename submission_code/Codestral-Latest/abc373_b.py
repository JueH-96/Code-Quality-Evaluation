# YOUR CODE HERE
import sys

def calculate_minimal_distance(S):
    # Create a dictionary to map each character to its position
    position = {char: index + 1 for index, char in enumerate(S)}

    # Start at the position of 'A'
    current_position = position['A']
    total_distance = 0

    # Calculate the distance for each character from 'B' to 'Z'
    for char in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
        next_position = position[char]
        total_distance += abs(next_position - current_position)
        current_position = next_position

    return total_distance

# Read input from stdin
S = sys.stdin.read().strip()

# Calculate and print the minimal possible total traveled distance
print(calculate_minimal_distance(S))