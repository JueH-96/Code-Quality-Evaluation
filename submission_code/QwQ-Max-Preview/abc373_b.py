# Read input
S = input().strip()

# Create a dictionary mapping each character to its 1-based position in S
positions = {char: idx + 1 for idx, char in enumerate(S)}

# List of characters in the order they need to be typed
characters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

total_distance = 0
current_pos = positions['A']

# Iterate from B to Z and accumulate the total distance
for char in characters[1:]:
    next_pos = positions[char]
    total_distance += abs(next_pos - current_pos)
    current_pos = next_pos

print(total_distance)