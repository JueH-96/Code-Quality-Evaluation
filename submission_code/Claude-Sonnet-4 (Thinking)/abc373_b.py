S = input().strip()

# Find the position of each letter (1-indexed)
pos = {}
for i, char in enumerate(S):
    pos[char] = i + 1

# Calculate total distance
total_distance = 0
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Calculate distance for each consecutive pair
for i in range(25):  # 25 transitions: A→B, B→C, ..., Y→Z
    current_char = alphabet[i]
    next_char = alphabet[i + 1]
    current_pos = pos[current_char]
    next_pos = pos[next_char]
    total_distance += abs(next_pos - current_pos)

print(total_distance)