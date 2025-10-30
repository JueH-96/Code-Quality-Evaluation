import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
S_x = int(data[2])
S_y = int(data[3])

houses = set()
for i in range(N):
    houses.add((int(data[4 + 2*i]), int(data[5 + 2*i])))

x, y = S_x, S_y
visited_houses = set()

for i in range(M):
    direction = data[4 + 2*N + 2*i]
    distance = int(data[5 + 2*N + 2*i])

    if direction == 'U':
        for j in range(1, distance + 1):
            if (x, y + j) in houses:
                visited_houses.add((x, y + j))
        y += distance
    elif direction == 'D':
        for j in range(1, distance + 1):
            if (x, y - j) in houses:
                visited_houses.add((x, y - j))
        y -= distance
    elif direction == 'L':
        for j in range(1, distance + 1):
            if (x - j, y) in houses:
                visited_houses.add((x - j, y))
        x -= distance
    elif direction == 'R':
        for j in range(1, distance + 1):
            if (x + j, y) in houses:
                visited_houses.add((x + j, y))
        x += distance

print(x, y, len(visited_houses))