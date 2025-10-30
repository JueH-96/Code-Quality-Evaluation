import sys

def main():
    n, h, w = map(int, sys.stdin.readline().split())
    tiles = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    required_area = h * w

    for mask in range(1 << n):
        subset = [tiles[i] for i in range(n) if (mask & (1 << i))]
        area = sum(a * b for a, b in subset)
        if area != required_area:
            continue

        possible_orientations = []
        for a, b in subset:
            orientations = []
            if a != b:
                orientations.append((a, b))
                orientations.append((b, a))
            else:
                orientations.append((a, b))
            possible_orientations.append(orientations)

        def can_cover(mask, tiles_remaining):
            if mask == (1 << (h * w)) - 1:
                return True

            for i in range(h):
                for j in range(w):
                    cell = i * w + j
                    if not (mask & (1 << cell)):
                        break
                else:
                    continue
                break

            else:
                return True

            for idx, tile in enumerate(tiles_remaining):
                for (a, b) in possible_orientations[idx]:
                    for x in range(h - a + 1):
                        for y in range(w - b + 1):
                            if i < x + a and j < y + b:
                                valid = True
                                for dx in range(a):
                                    for dy in range(b):
                                        cx = x + dx
                                        cy = y + dy
                                        cell = cx * w + cy
                                        if mask & (1 << cell):
                                            valid = False
                                            break
                                    if not valid:
                                        break
                                if valid:
                                    new_mask = mask
                                    for dx in range(a):
                                        for dy in range(b):
                                            cx = x + dx
                                            cy = y + dy
                                            cell = cx * w + cy
                                            new_mask |= (1 << cell)
                                    new_tiles_remaining = tiles_remaining[:idx] + tiles_remaining[idx+1:]
                                    if can_cover(new_mask, new_tiles_remaining):
                                        return True
            return False

        if can_cover(0, subset):
            print("Yes")
            return

    print("No")

if __name__ == "__main__":
    main()