n = int(input())
size = 3 ** n

for i in range(size):
    row = []
    for j in range(size):
        current_level = n
        x, y = i, j
        black = True
        while current_level > 0:
            block_size = 3 ** (current_level - 1)
            bx = x // block_size
            by = y // block_size
            if bx == 1 and by == 1:
                black = False
                break
            x %= block_size
            y %= block_size
            current_level -= 1
        row.append('#' if black else '.')
    print(''.join(row))