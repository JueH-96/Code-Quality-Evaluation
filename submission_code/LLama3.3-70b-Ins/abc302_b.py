import sys

def find_snuke(grid):
    """
    Find the positions of the cells with 's', 'n', 'u', 'k', and 'e' in the grid.

    Args:
        grid (list of str): A 2D grid of characters.

    Returns:
        list of tuple: A list of (row, col) positions of the cells with 's', 'n', 'u', 'k', and 'e'.
    """
    directions = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    def is_valid(row, col):
        return 0 <= row < len(grid) and 0 <= col < len(grid[0])

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 's':
                for direction in directions:
                    positions = [(row, col)]
                    for i in range(4):
                        next_row, next_col = positions[-1][0] + direction[0], positions[-1][1] + direction[1]
                        if not is_valid(next_row, next_col):
                            break
                        positions.append((next_row, next_col))
                    else:
                        if ''.join(grid[r][c] for r, c in positions) == 'snuke':
                            return positions

    return None

def main():
    H, W = map(int, input().split())
    grid = [input() for _ in range(H)]

    positions = find_snuke(grid)
    if positions:
        for row, col in positions:
            print(row + 1, col + 1)

if __name__ == '__main__':
    main()