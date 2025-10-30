from collections import defaultdict

def min_total_distance(keyboard_layout):
    # Create a dictionary to store the position of each key
    key_positions = defaultdict(int)
    for i, key in enumerate(keyboard_layout, 1):
        key_positions[key] = i

    # Initialize the current position to the position of 'A'
    current_position = key_positions['A']
    total_distance = 0

    # Iterate through the keys and calculate the total distance
    for key in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        total_distance += abs(current_position - key_positions[key])
        current_position = key_positions[key]

    return total_distance

# Read the input from stdin
keyboard_layout = input().strip()

# Solve the problem and print the answer
print(min_total_distance(keyboard_layout))