# YOUR CODE HERE
S = input().strip()
# Create a dictionary to map each character to its position
char_to_pos = {char: idx for idx, char in enumerate(S)}
# The order of characters to be pressed
order = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
total_distance = 0
# Initial position is the position of 'A'
current_pos = char_to_pos['A']
for char in order[1:]:
    next_pos = char_to_pos[char]
    total_distance += abs(next_pos - current_pos)
    current_pos = next_pos
print(total_distance)