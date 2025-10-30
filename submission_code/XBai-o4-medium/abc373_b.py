s = input().strip()

# Create a dictionary to map each character to its position (1-based index)
position = {char: i + 1 for i, char in enumerate(s)}

total_distance = 0

# Iterate through each consecutive pair in the standard alphabet order
for i in range(25):
    current_char = chr(ord('A') + i)
    next_char = chr(ord('A') + i + 1)
    total_distance += abs(position[current_char] - position[next_char])

print(total_distance)