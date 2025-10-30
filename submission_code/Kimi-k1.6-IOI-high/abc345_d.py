def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    H = int(input[idx])
    idx += 1
    W = int(input[idx])
    idx += 1
    tiles = []
    for _ in range(N):
        A = int(input[idx])
        B = int(input[idx + 1])
        tiles.append((A, B))
        idx += 2
    
    def find_first_empty(grid_mask):
        for i in range(H):
            for j in range(W):
                pos = i * W + j
                if not (grid_mask & (1 << pos)):
                    return (i, j)
        return None
    
    def backtrack(grid_mask, used_tiles):
        first_empty = find_first_empty(grid_mask)
        if first_empty is None:
            return True
        i, j = first_empty
        for k in range(N):
            if not (used_tiles & (1 << k)):
                a, b = tiles[k]
                orientations = [(a, b)]
                if a != b:
                    orientations.append((b, a))
                for h, w in orientations:
                    if (i + h > H) or (j + w > W):
                        continue
                    mask = 0
                    for x in range(i, i + h):
                        for y in range(j, j + w):
                            pos = x * W + y
                            mask |= 1 << pos
                    if (grid_mask & mask) != 0:
                        continue
                    new_grid = grid_mask | mask
                    new_used = used_tiles | (1 << k)
                    if backtrack(new_grid, new_used):
                        return True
        return False
    
    print("Yes" if backtrack(0, 0) else "No")

if __name__ == "__main__":
    main()