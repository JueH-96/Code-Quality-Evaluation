n = int(input())
chords = []
for _ in range(n):
    a, b = map(int, input().split())
    if a > b:
        a, b = b, a
    chords.append((a, b))

chords.sort()

max_b = 0
intersect = False

for s, e in chords:
    if max_b > s and e > max_b:
        intersect = True
        break
    if e > max_b:
        max_b = e

print("Yes" if intersect else "No")