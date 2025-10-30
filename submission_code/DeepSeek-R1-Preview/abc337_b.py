s = input().strip()

segments = []
current = s[0]
segments.append(current)

for c in s[1:]:
    if c != current:
        segments.append(c)
        current = c

valid = True
for i in range(1, len(segments)):
    if segments[i] < segments[i-1]:
        valid = False
        break

print("Yes" if valid else "No")