# YOUR CODE HERE
S = input().strip()

# Create a dictionary to store the position of each character
position = {}
for i in range(26):
    position[S[i]] = i + 1  # 1-indexed positions

# Calculate total distance
total_distance = 0
current_pos = position['A']  # Start at position of 'A'

# Type each letter from A to Z
for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
    next_pos = position[char]
    total_distance += abs(next_pos - current_pos)
    current_pos = next_pos

print(total_distance)