n, H, W = map(int, input().split())
tiles = [tuple(map(int, input().split())) for _ in range(n)]
target = H * W

def can_tile(subset):
    grid = [0] * H  # Each row is represented as a bitmask

    def backtrack(remaining, grid):
        # Find the first empty cell
        for x in range(H):
            row = grid[x]
            for y in range(W):
                if not (row & (1 << (W - 1 - y))):
                    # Try all remaining tiles in both orientations
                    for i in range(len(remaining)):
                        a, b = remaining[i]
                        for (h, w) in [(a, b), (b, a)]:
                            if h == 0 or w == 0:
                                continue
                            if x + h > H or y + w > W:
                                continue
                            # Check all required rows
                            valid = True
                            for dx in range(h):
                                mask = ((1 << w) - 1) << (W - (y + w))
                                if (grid[x + dx] & mask) != 0:
                                    valid = False
                                    break
                            if not valid:
                                continue
                            # Create new grid and remaining
                            new_grid = grid.copy()
                            for dx in range(h):
                                mask = ((1 << w) - 1) << (W - (y + w))
                                new_grid[x + dx] |= mask
                            new_remaining = remaining[:i] + remaining[i+1:]
                            if backtrack(new_remaining, new_grid):
                                return True
                    # No tile fits here
                    return False
        # All cells filled
        return True

    return backtrack(subset, grid)

# Iterate over all possible subsets of tiles
for mask in range(1, 1 << n):
    subset = []
    total_area = 0
    for i in range(n):
        if mask & (1 << i):
            a, b = tiles[i]
            subset.append((a, b))
            total_area += a * b
    if total_area != target:
        continue
    if can_tile(subset):
        print("Yes")
        exit()

print("No")