import string

# Read the keyboard layout string S
S = input()

# Create a dictionary to map each character to its 1-indexed coordinate
char_to_pos = {}
for i, char_in_S in enumerate(S):
    char_to_pos[char_in_S] = i + 1
# A more compact way using dict comprehension:
# char_to_pos = {char: i + 1 for i, char in enumerate(S)}

# The sequence of characters to type is "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
typing_sequence = string.ascii_uppercase

# Initialize total traveled distance
total_distance = 0

# Initially, the finger is at the coordinate of the key for 'A'.
# After 'A' is pressed, the finger is at this position.
# This will be the starting point for the first move (A to B).
current_finger_actual_pos = char_to_pos[typing_sequence[0]] # Position of 'A'

# Iterate through the characters to be typed, from the second one ('B') to the last one ('Z')
# For each character, calculate the distance moved from the previous character's key
# and update the finger's current position.
for i in range(1, len(typing_sequence)):
    # Get the next character in alphabetical order (e.g., 'B', then 'C', ...)
    next_char_in_alphabetical_order = typing_sequence[i]
    
    # Find the coordinate of the key for this next character
    pos_of_key_for_next_char = char_to_pos[next_char_in_alphabetical_order]
    
    # Calculate the distance moved in this step
    distance_for_this_step = abs(pos_of_key_for_next_char - current_finger_actual_pos)
    
    # Add this step's distance to the total
    total_distance += distance_for_this_step
    
    # Update the finger's current position to the key just pressed
    current_finger_actual_pos = pos_of_key_for_next_char
    
# Print the total traveled distance
print(total_distance)