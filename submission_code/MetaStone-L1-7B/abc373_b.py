S = input().strip()

positions = {}
for i in range(26):
    c = S[i]
    positions[c] = i + 1

order = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
pos_list = [positions[c] for c in order]

total = 0
for i in range(25):
    total += abs(pos_list[i+1] - pos_list[i])

print(total)