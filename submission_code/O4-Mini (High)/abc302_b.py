import sys

def main():
    input = sys.stdin.readline
    H, W = map(int, input().split())
    grid = [input().rstrip() for _ in range(H)]
    target = "snuke"
    # All 8 possible directions: vertical, horizontal, diagonal
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]
    
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 's':
                continue
            # Try each direction
            for dr, dc in directions:
                coords = []
                ok = True
                # Check the 5 letters in this direction
                for i, ch in enumerate(target):
                    rr = r + dr * i
                    cc = c + dc * i
                    if 0 <= rr < H and 0 <= cc < W and grid[rr][cc] == ch:
                        coords.append((rr + 1, cc + 1))  # 1-based
                    else:
                        ok = False
                        break
                if ok:
                    # Found the unique "snuke" sequence
                    for rr, cc in coords:
                        print(rr, cc)
                    return

if __name__ == "__main__":
    main()