import sys

def can_tile(tiles, H, W):
    grid = [[False for _ in range(W)] for _ in range(H)]
    # Sort tiles by descending area, then descending max dimension
    tiles = sorted(tiles, key=lambda x: (-x[0]*x[1], -max(x[0], x[1])))

    def backtrack(remaining):
        # Find first empty cell
        for r in range(H):
            for c in range(W):
                if not grid[r][c]:
                    break
            else:
                continue
            break
        else:
            # All cells filled
            return True

        # Try each tile in remaining
        for i in range(len(remaining)):
            tile = remaining[i]
            a, b = tile
            # Try both orientations
            for h, w in [(a, b), (b, a)]:
                if r + h > H or c + w > W:
                    continue
                # Check if the rectangle is free
                collision = False
                for x in range(r, r + h):
                    for y in range(c, c + w):
                        if grid[x][y]:
                            collision = True
                            break
                    if collision:
                        break
                if collision:
                    continue
                # Place the tile
                for x in range(r, r + h):
                    for y in range(c, c + w):
                        grid[x][y] = True
                new_remaining = remaining[:i] + remaining[i+1:]
                if backtrack(new_remaining):
                    return True
                # Unplace the tile
                for x in range(r, r + h):
                    for y in range(c, c + w):
                        grid[x][y] = False
        return False

    return backtrack(tiles)

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    H = int(input[idx]); idx +=1
    W = int(input[idx]); idx +=1
    tiles = []
    for _ in range(N):
        a = int(input[idx]); idx +=1
        b = int(input[idx]); idx +=1
        tiles.append( (a, b) )
    target = H * W
    for mask in range(1, 1 << N):
        subset = []
        for i in range(N):
            if mask & (1 << i):
                subset.append( tiles[i] )
        sum_area = sum( a*b for a, b in subset )
        if sum_area != target:
            continue
        # Check if all tiles can fit in some orientation
        valid = True
        for a, b in subset:
            if not ( (a <= H and b <= W) or (b <= H and a <= W) ):
                valid = False
                break
        if not valid:
            continue
        if can_tile(subset, H, W):
            print("Yes")
            return
    print("No")

if __name__ == "__main__":
    main()