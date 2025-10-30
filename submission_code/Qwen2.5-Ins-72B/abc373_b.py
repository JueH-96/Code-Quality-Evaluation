# YOUR CODE HERE
S = input().strip()
key_positions = {char: idx for idx, char in enumerate(S, start=1)}
current_position = key_positions['A']
total_distance = 0

for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    next_position = key_positions[char]
    total_distance += abs(next_position - current_position)
    current_position = next_position

print(total_distance)