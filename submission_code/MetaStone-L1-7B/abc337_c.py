n = int(input())
a = list(map(int, input().split()))
front = -1
for i in range(n):
    if a[i] == -1:
        front = i + 1
        break

next_person = [0] * (n + 2)  # 1-based indexing for people

for i in range(n):
    ai = a[i]
    if ai != -1:
        next_person[ai] = i + 1

res = []
current = front
while current != 0:
    res.append(str(current))
    current = next_person[current]

print(' '.join(res))