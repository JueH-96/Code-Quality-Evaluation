s = input().strip()
t = input().strip()

i = 0
positions = []
for j in range(len(t)):
    if i < len(s) and t[j] == s[i]:
        positions.append(j + 1)
        i += 1

print(' '.join(map(str, positions)))