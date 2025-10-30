import sys

def main():
    import sys

    lines = sys.stdin.read().splitlines()
    if not lines:
        return

    # Parse first line
    first_line = lines[0].split()
    H, W, X, Y = map(int, first_line)

    # Read grid lines
    grid = lines[1:H+1]

    # Read movement string
    T = lines[H+1].strip()

    # Current position
    x, y = X, Y

    # Set to keep track of passed houses
    passed_houses = set()

    # Direction mappings
    directions = {
        'U': (-1, 0),
        'D': (1, 0),
        'L': (0, -1),
        'R': (0, 1)
    }

    for move in T:
        if move not in directions:
            continue  # Ignore invalid directions if any

        dx, dy = directions[move]
        new_x = x + dx
        new_y = y + dy

        # Check if the new position is within bounds and passable
        if 1 <= new_x <= H and 1 <= new_y <= W:
            if grid[new_x - 1][new_y - 1] != '#':
                x, y = new_x, new_y
                if grid[x - 1][y - 1] == '@':
                    passed_houses.add((x, y))

    # Output the final position and the number of distinct houses passed
    print(x, y, len(passed_houses))

if __name__ == "__main__":
    main()