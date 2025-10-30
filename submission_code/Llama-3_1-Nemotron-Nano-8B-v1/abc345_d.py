import itertools

n, h, w = map(int, input().split())
tiles = [tuple(map(int, input().split())) for _ in range(n)]
total_cells = h * w

for choices in itertools.product([0, 1, 2], repeat=n):
    selected = []
    sum_area = 0
    for i in range(n):
        c = choices[i]
        if c == 0:
            continue
        a, b = tiles[i]
        if c == 1:
            selected.append((a, b))
        else:
            selected.append((b, a))
        sum_area += a * b
    if sum_area != total_cells:
        continue
    selected.sort(key=lambda x: (-(x[0] * x[1]), x[0], x[1]))
    
    def backtrack(grid_mask, idx):
        if idx == len(selected):
            return grid_mask == (1 << total_cells) - 1
        a, b = selected[idx]
        remaining = total_cells - bin(grid_mask).count('1')
        if a * b > remaining:
            return False
        max_x = h - a
        max_y = w - b
        for x in range(max_x + 1):
            for y in range(max_y + 1):
                tile_mask = 0
                for i in range(a):
                    for j in range(b):
                        pos = (x + i) * w + (y + j)
                        tile_mask |= 1 << pos
                if (grid_mask & tile_mask) != 0:
                    continue
                new_mask = grid_mask | tile_mask
                if backtrack(new_mask, idx + 1):
                    return True
        return False
    
    if backtrack(0, 0):
        print("Yes")
        exit()

print("No")