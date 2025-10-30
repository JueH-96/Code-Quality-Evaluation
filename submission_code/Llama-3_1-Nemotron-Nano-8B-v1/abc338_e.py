n = int(input())
chords = []
two_n = 2 * n

for _ in range(n):
    a, b = map(int, input().split())
    distance = (b - a) % two_n
    if distance <= n:
        s, t = a, b
    else:
        s, t = b, a
    if s > t:
        t += two_n
    chords.append((s, t))

chords.sort()
max_end = 0

for s, t in chords:
    if max_end > s and max_end < t:
        print("Yes")
        exit()
    if t > max_end:
        max_end = t

print("No")