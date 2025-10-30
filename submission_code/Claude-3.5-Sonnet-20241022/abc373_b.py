S = input().strip()

# Create a mapping of character to position
pos = {char: i+1 for i, char in enumerate(S)}

# Calculate total distance
total_distance = 0
current_pos = pos['A']  # Start at A

# Move through alphabet in order
for c in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
    next_pos = pos[c]
    distance = abs(next_pos - current_pos)
    total_distance += distance
    current_pos = next_pos

print(total_distance)