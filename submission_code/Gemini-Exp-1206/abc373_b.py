s = input()
pos = {}
for i in range(26):
    pos[s[i]] = i + 1
dist = 0
curr = pos['A']
for i in range(ord('A'), ord('Z') + 1):
    dist += abs(pos[chr(i)] - curr)
    curr = pos[chr(i)]
print(dist)