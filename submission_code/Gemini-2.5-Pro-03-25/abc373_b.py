# YOUR CODE HERE
import sys

# Read the input string S which defines the keyboard layout
# S is a permutation of the 26 uppercase English letters.
s = sys.stdin.readline().strip()

# Create a dictionary to map each character to its coordinate on the number line.
# The problem states that the key for character S_x is at coordinate x (1 <= x <= 26).
# In 0-based indexing for the string S, the character S[i] corresponds to S_{i+1}.
# Therefore, the character S[i] is located at coordinate i + 1.
char_to_pos = {}
for i, char in enumerate(s):
    # Store the mapping: character -> coordinate
    char_to_pos[char] = i + 1
    
# Initialize the total traveled distance to 0.
total_distance = 0

# Define the sequence of characters to be typed in alphabetical order: A, B, C, ..., Z.
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# The finger starts at the coordinate of 'A'. Pressing 'A' requires no movement initially.
# We need to calculate the total distance traveled for the subsequent key presses from 'B' to 'Z'.
# The loop iterates through the sequence from the second character ('B') up to the last ('Z').
for i in range(1, 26):
    # Get the current character to type (e.g., 'B', then 'C', ...)
    current_char = alphabet[i]
    # Get the previously typed character (e.g., 'A', then 'B', ...)
    previous_char = alphabet[i-1]
    
    # Look up the coordinates of these two characters using the precomputed map.
    pos_current = char_to_pos[current_char]
    pos_previous = char_to_pos[previous_char]
    
    # Calculate the distance moved for this step.
    # The distance between two coordinates on a number line is the absolute difference.
    distance = abs(pos_current - pos_previous)
    
    # Add this step's distance to the running total distance.
    total_distance += distance
    
# After iterating through all the steps from A to Z, print the final total distance.
print(total_distance)