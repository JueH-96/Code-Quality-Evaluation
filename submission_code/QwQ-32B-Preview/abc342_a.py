S = input().strip()

frequency = {}

for char in S:
    if char in frequency:
        frequency[char] += 1
    else:
        frequency[char] = 1

# Find the character with count 1
for char, count in frequency.items():
    if count == 1:
        diff_char = char
        break

# Find its position
position = S.index(diff_char) + 1

print(position)