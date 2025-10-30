import sys

def calculate_distance(keyboard_layout):
    # Find the initial position of the finger (position of 'A')
    initial_position = keyboard_layout.index('A') + 1
    total_distance = 0
    
    # Calculate the distance for each character from A to Z
    for i in range(1, 26):
        current_char = chr(ord('A') + i)
        current_position = keyboard_layout.index(current_char) + 1
        total_distance += abs(current_position - initial_position)
        initial_position = current_position
    
    return total_distance

# Read input from stdin
keyboard_layout = sys.stdin.readline().strip()

# Calculate and print the minimal possible total traveled distance
print(calculate_distance(keyboard_layout))