def place_dragon_parts(N):
    # Initialize the grid with zeros
    grid = [[0] * N for _ in range(N)]

    # Place Takahashi at the center
    center = N // 2
    grid[center][center] = 'T'

    # Fill the grid with dragon parts in a spiral pattern
    num = 1
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
    x, y = 0, 0
    direction_index = 0

    while num <= N * N - 1:
        grid[x][y] = num
        num += 1

        # Calculate the next position
        next_x = x + directions[direction_index][0]
        next_y = y + directions[direction_index][1]

        # Check if the next position is out of bounds or already filled
        if (next_x < 0 or next_x >= N or next_y < 0 or next_y >= N or grid[next_x][next_y] != 0):
            direction_index = (direction_index + 1) % 4
            next_x = x + directions[direction_index][0]
            next_y = y + directions[direction_index][1]

        x, y = next_x, next_y

    # Print the grid
    for row in grid:
        print(' '.join(map(str, row)))

# Read input
N = int(input().strip())

# Place dragon parts and print the grid
place_dragon_parts(N)