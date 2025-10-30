import sys
input = sys.stdin.read
data = input().split()

index = 0
N = int(data[index])
M = int(data[index + 1])
S_x = int(data[index + 2])
S_y = int(data[index + 3])
index += 4

houses = set()
for _ in range(N):
    X = int(data[index])
    Y = int(data[index + 1])
    houses.add((X, Y))
    index += 2

current_x, current_y = S_x, S_y
visited_houses = set()
visited_houses.add((current_x, current_y))

for _ in range(M):
    D = data[index]
    C = int(data[index + 1])
    index += 2

    if D == 'U':
        new_y = current_y + C
        for y in range(current_y + 1, new_y + 1):
            if (current_x, y) in houses:
                visited_houses.add((current_x, y))
        current_y = new_y
    elif D == 'D':
        new_y = current_y - C
        for y in range(current_y - 1, new_y - 1, -1):
            if (current_x, y) in houses:
                visited_houses.add((current_x, y))
        current_y = new_y
    elif D == 'L':
        new_x = current_x - C
        for x in range(current_x - 1, new_x - 1, -1):
            if (x, current_y) in houses:
                visited_houses.add((x, current_y))
        current_x = new_x
    elif D == 'R':
        new_x = current_x + C
        for x in range(current_x + 1, new_x + 1):
            if (x, current_y) in houses:
                visited_houses.add((x, current_y))
        current_x = new_x

print(current_x, current_y, len(visited_houses))