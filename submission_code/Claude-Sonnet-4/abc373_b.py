# YOUR CODE HERE
S = input().strip()

# Create a mapping from character to its position (1-indexed)
char_to_pos = {}
for i, char in enumerate(S):
    char_to_pos[char] = i + 1

# Start at position of 'A'
current_pos = char_to_pos['A']
total_distance = 0

# Type each letter from A to Z
for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    target_pos = char_to_pos[char]
    # Add distance to move from current position to target position
    total_distance += abs(target_pos - current_pos)
    # Update current position
    current_pos = target_pos

print(total_distance)