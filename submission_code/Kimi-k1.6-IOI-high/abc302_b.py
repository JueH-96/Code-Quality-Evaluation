h, w = map(int, input().split())
grid = [input().strip() for _ in range(h)]

directions = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]

found = False
result = []

for i in range(1, h + 1):
    for j in range(1, w + 1):
        if grid[i-1][j-1] == 's':
            for dr, dc in directions:
                valid = True
                for step in range(1, 5):
                    ni = i + dr * step
                    nj = j + dc * step
                    if ni < 1 or ni > h or nj < 1 or nj > w:
                        valid = False
                        break
                    required = ['n', 'u', 'k', 'e'][step-1]
                    if grid[ni-1][nj-1] != required:
                        valid = False
                        break
                if valid:
                    result = [(i + dr*st, j + dc*st) for st in range(5)]
                    found = True
                    break
            if found:
                break
        if found:
            break
    if found:
        break

for r, c in result:
    print(r, c)