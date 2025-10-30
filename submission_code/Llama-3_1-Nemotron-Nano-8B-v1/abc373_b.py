s = input().strip()
pos = {c: i + 1 for i, c in enumerate(s)}
total = 0
for i in range(1, 26):
    prev_char = chr(ord('A') + i - 1)
    curr_char = chr(ord('A') + i)
    total += abs(pos[curr_char] - pos[prev_char])
print(total)