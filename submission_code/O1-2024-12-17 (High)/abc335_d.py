def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])

    # Generate all cells in a spiral order (row, col), 0-based
    def spiral_coords(n):
        coords = []
        top, bottom = 0, n - 1
        left, right = 0, n - 1
        while True:
            # Move left to right on the top row
            for c in range(left, right + 1):
                coords.append((top, c))
            top += 1
            if top > bottom:
                break

            # Move top to bottom on the right column
            for r in range(top, bottom + 1):
                coords.append((r, right))
            right -= 1
            if left > right:
                break

            # Move right to left on the bottom row
            for c in range(right, left - 1, -1):
                coords.append((bottom, c))
            bottom -= 1
            if top > bottom:
                break

            # Move bottom to top on the left column
            for r in range(bottom, top - 1, -1):
                coords.append((r, left))
            left += 1
            if left > right:
                break
        return coords

    path = spiral_coords(N)  # This will be a continuous path of length N^2
    center = path[-1]        # The last cell in the spiral is the center
    path = path[:-1]         # Remove the center cell from the path

    # Prepare an N x N grid to place the parts
    grid = [[""] * N for _ in range(N)]

    # Number the path cells from 1 to N^2 - 1
    for idx, (r, c) in enumerate(path):
        grid[r][c] = str(idx + 1)

    # Place 'T' in the center
    cr, cc = center
    grid[cr][cc] = "T"

    # Print the grid
    for r in range(N):
        print(" ".join(grid[r]))

# Do not forget to call main()!
main()