import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
H = int(data[index])
index += 1
K = int(data[index])
index += 1
S = data[index]
index += 1

items = set()
for _ in range(M):
    x = int(data[index])
    index += 1
    y = int(data[index])
    index += 1
    items.add((x, y))

x, y = 0, 0
for move in S:
    if move == 'R':
        x += 1
    elif move == 'L':
        x -= 1
    elif move == 'U':
        y += 1
    elif move == 'D':
        y -= 1
    H -= 1
    if H < 0:
        print("No")
        break
    if (x, y) in items and H < K:
        H = K
else:
    print("Yes")