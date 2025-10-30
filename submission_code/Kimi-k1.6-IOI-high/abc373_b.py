s = input().strip()
positions = {char: idx + 1 for idx, char in enumerate(s)}
current = positions['A']
total = 0
for char in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
    next_pos = positions[char]
    total += abs(next_pos - current)
    current = next_pos
print(total)