def minimal_travel_distance(S):
    # Create a mapping of each character to its position
    position = {char: idx + 1 for idx, char in enumerate(S)}
    
    # Initialize the starting position for 'A'
    current_position = position['A']
    total_distance = 0
    
    # Calculate the distance for each character from A to Z
    for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        next_position = position[char]
        total_distance += abs(next_position - current_position)
        current_position = next_position
    
    return total_distance

# Read input
import sys
input = sys.stdin.read
S = input().strip()

# Output the result
print(minimal_travel_distance(S))