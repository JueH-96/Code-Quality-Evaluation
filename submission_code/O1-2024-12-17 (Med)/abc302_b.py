def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    H, W = map(int, input_data[:2])
    grid = input_data[2:]

    # Directions (dr, dc) for all 8 directions around a cell
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    target = "snuke"

    # Check within grid bounds
    def in_bounds(r, c):
        return 0 <= r < H and 0 <= c < W

    for r in range(H):
        for c in range(W):
            if grid[r][c] == 's':
                for dr, dc in directions:
                    # Check if we can follow the direction for the next letters
                    rr, cc = r, c
                    matched = True
                    for i in range(5):
                        if not in_bounds(rr, cc) or grid[rr][cc] != target[i]:
                            matched = False
                            break
                        rr += dr
                        cc += dc
                    if matched:
                        # We found the unique solution; output and return
                        rr, cc = r, c
                        for i in range(5):
                            print(rr+1, cc+1)
                            rr += dr
                            cc += dc
                        return

# Don't forget to call main()
if __name__ == "__main__":
    main()