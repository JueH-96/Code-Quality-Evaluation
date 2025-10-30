n = int(input())
a = list(map(int, input().split()))
front = a.index(-1) + 1
next_person = {}
for s in range(1, n + 1):
    ai = a[s - 1]
    if ai != -1:
        next_person[ai] = s
line = [front]
current = front
while current in next_person:
    current = next_person[current]
    line.append(current)
print(' '.join(map(str, line)))