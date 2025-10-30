S = input().strip()

# Create a mapping from each letter to its position
pos = {}
for idx, letter in enumerate(S, start=1):
    pos[letter] = idx

# Initialize total distance
total_distance = 0

# Starting position is 'A'
current_pos = pos['A']

# Iterate through B to Z
for letter in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
    next_pos = pos[letter]
    total_distance += abs(next_pos - current_pos)
    current_pos = next_pos

print(total_distance)