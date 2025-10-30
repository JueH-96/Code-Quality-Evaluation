S = input().strip()
position_map = {char: idx + 1 for idx, char in enumerate(S)}
total_distance = 0
previous_pos = position_map['A']
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    current_pos = position_map[char]
    total_distance += abs(current_pos - previous_pos)
    previous_pos = current_pos
print(total_distance)