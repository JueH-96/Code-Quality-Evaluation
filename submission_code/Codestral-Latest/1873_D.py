import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
results = []

for _ in range(t):
    n = int(data[index])
    k = int(data[index + 1])
    s = data[index + 2]
    index += 3

    black_cells = [i for i, cell in enumerate(s) if cell == 'B']
    if not black_cells:
        results.append(0)
        continue

    operations = 0
    i = 0
    while i < len(black_cells):
        start = black_cells[i]
        end = start
        while i < len(black_cells) and black_cells[i] < start + k:
            end = black_cells[i]
            i += 1
        operations += 1
        i = black_cells.index(end) + 1

    results.append(operations)

for result in results:
    print(result)