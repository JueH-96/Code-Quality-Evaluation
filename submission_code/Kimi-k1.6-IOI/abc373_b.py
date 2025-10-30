s = input().strip()
pos = {char: idx + 1 for idx, char in enumerate(s)}
sequence = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
total = 0
prev = pos['A']
for c in sequence[1:]:
    current = pos[c]
    total += abs(current - prev)
    prev = current
print(total)