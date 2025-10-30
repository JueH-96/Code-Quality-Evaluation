S = input().strip()
positions = {c: i + 1 for i, c in enumerate(S)}
total = 0
current_pos = positions['A']
for c in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
    next_pos = positions[c]
    total += abs(next_pos - current_pos)
    current_pos = next_pos
print(total)