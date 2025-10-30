import sys

# Read the keyboard layout string S
S = sys.stdin.readline().strip()

# Create a dictionary mapping characters to their coordinates (1-based)
char_to_coord = {}
for i in range(len(S)):
    char = S[i]
    coord = i + 1
    char_to_coord[char] = coord

# Define the sequence of characters to type
target_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Initialize total distance
total_distance = 0

# Get the starting coordinate (position of 'A')
# The finger is initially AT the key for 'A'. No distance traveled yet.
current_coord = char_to_coord['A']

# Iterate through the characters to type from the second character onwards (B to Z)
# For each character, calculate the distance moved from the previous character's position
for i in range(1, len(target_chars)):
    next_char = target_chars[i]
    next_coord = char_to_coord[next_char]

    # Calculate the distance moved to reach the next key
    distance = abs(next_coord - current_coord)

    # Add to total distance
    total_distance += distance

    # Update current position for the next iteration
    current_coord = next_coord

# Print the total distance
print(total_distance)