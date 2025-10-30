import sys

def main() -> None:
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().rstrip() for _ in range(H)]

    min_r, max_r = H, -1
    min_c, max_c = W, -1

    # find the bounding rectangle that still contains '#'
    for r in range(H):
        for c in range(W):
            if grid[r][c] == '#':
                if r < min_r:
                    min_r = r
                if r > max_r:
                    max_r = r
                if c < min_c:
                    min_c = c
                if c > max_c:
                    max_c = c

    # inside that rectangle exactly one cell is '.'
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r][c] == '.':
                # convert to 1-based indices for the answer
                print(r + 1, c + 1)
                return

if __name__ == "__main__":
    main()