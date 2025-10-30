# YOUR CODE HERE
S = input().strip()

# Find the position (1-indexed) of each character
position = {}
for i in range(26):
    position[S[i]] = i + 1

# Calculate total distance
total_distance = 0
current_pos = position['A']

for char in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
    next_pos = position[char]
    total_distance += abs(next_pos - current_pos)
    current_pos = next_pos

print(total_distance)