s = input().strip()
pos = {c: i+1 for i, c in enumerate(s)}
current = pos['A']
total = 0
for c in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
    next_pos = pos[c]
    total += abs(next_pos - current)
    current = next_pos
print(total)