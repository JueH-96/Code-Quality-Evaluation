import itertools

c = []
for _ in range(3):
    c.append(list(map(int, input().split())))

grid = []
for i in range(3):
    for j in range(3):
        grid.append((i, j))

count = 0
total = 0
for perm in itertools.permutations(grid):
    total += 1
    disappointed = False
    seen = [[False] * 3 for _ in range(3)]
    for i, j in perm:
        seen[i][j] = True
        lines = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(0, 2), (1, 1), (2, 0)],
        ]
        for line in lines:
            first = line[0]
            second = line[1]
            third = line[2]
            if seen[first[0]][first[1]] and seen[second[0]][second[1]] and seen[third[0]][third[1]]:
                if c[first[0]][first[1]] == c[second[0]][second[1]] and c[first[0]][first[1]] != c[third[0]][third[1]]:
                    disappointed = True
                    break
        if disappointed:
            break
    if not disappointed:
        count += 1

print(count / total)