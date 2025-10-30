import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
H = int(data[2])
K = int(data[3])
S = data[4]

items = set()
for i in range(M):
    x, y = int(data[5 + 2 * i]), int(data[6 + 2 * i])
    items.add((x, y))

x, y = 0, 0
health = H

for move in S:
    if move == 'R':
        x += 1
    elif move == 'L':
        x -= 1
    elif move == 'U':
        y += 1
    elif move == 'D':
        y -= 1

    health -= 1
    if health < 0:
        print("No")
        break

    if (x, y) in items and health < K:
        health = K
        items.remove((x, y))
else:
    print("Yes")