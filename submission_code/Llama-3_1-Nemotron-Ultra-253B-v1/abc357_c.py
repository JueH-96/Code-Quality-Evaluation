def generate(k):
    if k == 0:
        return ['#']
    prev = generate(k-1)
    size = len(prev)
    new_grid = []
    for block_row in [0, 1, 2]:
        for i in range(size):
            row = []
            for block_col in [0, 1, 2]:
                if block_row == 1 and block_col == 1:
                    row.append('.' * size)
                else:
                    row.append(prev[i])
            new_grid.append(''.join(row))
    return new_grid

n = int(input())
carpet = generate(n)
for line in carpet:
    print(line)