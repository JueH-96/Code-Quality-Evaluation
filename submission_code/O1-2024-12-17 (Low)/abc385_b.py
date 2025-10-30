def main():
    import sys

    data = sys.stdin.read().strip().split()
    H, W, X, Y = map(int, data[:4])
    grid = data[4:4+H]
    T = data[4+H]

    # Convert to 0-based for internal convenience
    # But we'll store them in a structure that we can index easily.
    # Note that X,Y are 1-based in the problem statement, so we shift them.
    X -= 1
    Y -= 1

    # Check passability function
    def is_passable(r, c):
        return grid[r][c] in ('.', '@')

    # Keep track of visited houses
    visited_houses = set()

    # Function to try moving
    # If passable, move, otherwise stay
    # After moving, if on a house, add to visited set
    def move(pos_x, pos_y, direction):
        if direction == 'U':
            nx, ny = pos_x - 1, pos_y
        elif direction == 'D':
            nx, ny = pos_x + 1, pos_y
        elif direction == 'L':
            nx, ny = pos_x, pos_y - 1
        elif direction == 'R':
            nx, ny = pos_x, pos_y + 1
        else:
            nx, ny = pos_x, pos_y

        if is_passable(nx, ny):
            return nx, ny
        return pos_x, pos_y

    # Simulate
    cur_x, cur_y = X, Y
    for ch in T:
        # Move if possible
        cur_x, cur_y = move(cur_x, cur_y, ch)
        # After possibly moving, check if house
        if grid[cur_x][cur_y] == '@':
            visited_houses.add((cur_x, cur_y))

    # Convert back to 1-based for output
    print(cur_x+1, cur_y+1, len(visited_houses))

if __name__ == "__main__":
    main()