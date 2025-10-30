s = input().strip()
mapping = {char: idx + 1 for idx, char in enumerate(s)}
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
current = mapping['A']
total = 0
for char in alphabet[1:]:
	next_pos = mapping[char]
	total += abs(next_pos - current)
	current = next_pos
print(total)