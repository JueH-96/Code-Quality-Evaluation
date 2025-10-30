import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    H = int(data[0])
    W = int(data[1])
    grid = data[2:]

    min_r = H
    max_r = -1
    min_c = W
    max_c = -1

    # Find bounding box of existing black cells '#'
    for i in range(H):
        row = grid[i]
        for j, ch in enumerate(row):
            if ch == '#':
                if i < min_r: min_r = i
                if i > max_r: max_r = i
                if j < min_c: min_c = j
                if j > max_c: max_c = j

    # Problem guarantees at least one '#', but just in case:
    if max_r == -1:
        # No black cells: but problem says there's at least one, so this won't happen.
        print("Yes")
        return

    # Check every cell in the bounding rectangle
    # If any is '.', it can't be painted black => impossible
    for i in range(min_r, max_r + 1):
        row = grid[i]
        # slice of interest: cols min_c..max_c
        for j in range(min_c, max_c + 1):
            if row[j] == '.':
                print("No")
                return

    # Otherwise we can paint all '?' inside to '#' and all outside to '.'
    print("Yes")


if __name__ == "__main__":
    main()