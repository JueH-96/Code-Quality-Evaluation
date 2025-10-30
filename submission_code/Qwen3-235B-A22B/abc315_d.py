import sys

def main():
    H, W = map(int, sys.stdin.readline().split())
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    alive = [[True for _ in range(W)] for _ in range(H)]

    while True:
        marked = [[False for _ in range(W)] for _ in range(H)]
        changed = False

        # Check rows
        for i in range(H):
            first_char = None
            valid = True
            count = 0
            for j in range(W):
                if alive[i][j]:
                    if first_char is None:
                        first_char = grid[i][j]
                        count = 1
                    else:
                        if grid[i][j] != first_char:
                            valid = False
                            break
                        else:
                            count += 1
            if valid and count >= 2:
                # Mark all alive cells in this row
                for j in range(W):
                    if alive[i][j]:
                        marked[i][j] = True
                changed = True

        # Check columns
        for j in range(W):
            first_char = None
            valid = True
            count = 0
            for i in range(H):
                if alive[i][j]:
                    if first_char is None:
                        first_char = grid[i][j]
                        count = 1
                    else:
                        if grid[i][j] != first_char:
                            valid = False
                            break
                        count += 1
            if valid and count >= 2:
                # Mark all alive cells in this column
                for i in range(H):
                    if alive[i][j]:
                        marked[i][j] = True
                changed = True

        if not changed:
            break

        # Remove marked cells
        for i in range(H):
            for j in range(W):
                if marked[i][j]:
                    alive[i][j] = False

    # Count remaining cookies
    total = 0
    for row in alive:
        total += sum(row)
    print(total)

if __name__ == "__main__":
    main()