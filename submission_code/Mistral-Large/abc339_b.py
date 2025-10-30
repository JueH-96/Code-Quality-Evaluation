import sys

def main():
    input = sys.stdin.read
    data = input().split()

    H = int(data[0])
    W = int(data[1])
    N = int(data[2])

    # Initialize the grid with all white cells
    grid = [['.' for _ in range(W)] for _ in range(H)]

    # Directions: 0 -> Up, 1 -> Right, 2 -> Down, 3 -> Left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    current_direction = 0
    current_position = (0, 0)

    for _ in range(N):
        x, y = current_position

        # Toggle the color of the current cell
        if grid[x][y] == '.':
            grid[x][y] = '#'
            current_direction = (current_direction + 1) % 4
        else:
            grid[x][y] = '.'
            current_direction = (current_direction - 1) % 4

        # Move to the next cell
        dx, dy = directions[current_direction]
        current_position = ((x + dx) % H, (y + dy) % W)

    # Print the final grid
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()