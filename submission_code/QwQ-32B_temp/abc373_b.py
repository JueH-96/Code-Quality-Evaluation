s = input().strip()
pos = {c: i + 1 for i, c in enumerate(s)}
order = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
total = 0
current = pos[order[0]]
for i in range(1, 26):
    next_char = order[i]
    next_p = pos[next_char]
    total += abs(next_p - current)
    current = next_p
print(total)