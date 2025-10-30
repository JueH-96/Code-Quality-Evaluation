import sys

# Read input
line = sys.stdin.readline().split()
N, M, Sx, Sy = map(int, line)
houses = []
for _ in range(N):
    line = sys.stdin.readline().split()
    x, y = map(int, line)
    houses.append((x, y))
actions = []
for _ in range(M):
    line = sys.stdin.readline().split()
    d, c = line[0], int(line[1])
    actions.append((d, c))

# Simulate Santa's movements
x, y = Sx, Sy
visited = set()
for d, c in actions:
    if d == 'U':
        for i in range(1, c+1):
            if (x, y+i) in houses:
                visited.add((x, y+i))
        y += c
    elif d == 'D':
        for i in range(1, c+1):
            if (x, y-i) in houses:
                visited.add((x, y-i))
        y -= c
    elif d == 'L':
        for i in range(1, c+1):
            if (x-i, y) in houses:
                visited.add((x-i, y))
        x -= c
    else:
        for i in range(1, c+1):
            if (x+i, y) in houses:
                visited.add((x+i, y))
        x += c

# Print the result
print(x, y, len(visited))