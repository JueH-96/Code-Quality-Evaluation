# YOUR CODE HERE
S = input().strip()
# Create a dictionary to map each character to its position
char_to_pos = {char: idx for idx, char in enumerate(S)}
# Initialize the total distance
total_distance = 0
# Start at the position of 'A'
current_pos = char_to_pos['A']
# Iterate through the alphabet from 'B' to 'Z'
for char in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
    next_pos = char_to_pos[char]
    # Calculate the absolute difference in positions
    distance = abs(next_pos - current_pos)
    total_distance += distance
    # Update the current position
    current_pos = next_pos
print(total_distance)