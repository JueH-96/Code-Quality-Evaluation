import itertools
import sys

def is_disappointed(grid, perm):
    for line in [
        [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],  # Rows
        [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],  # Columns
        [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]  # Diagonals
    ]:
        indices = [perm.index(grid[i][j]) for i, j in line]
        if indices[0] < indices[1] < indices[2] and grid[line[0][0]][line[0][1]] == grid[line[1][0]][line[1][1]] != grid[line[2][0]][line[2][1]]:
            return True
    return False

def main():
    input = sys.stdin.read
    data = input().strip().split()
    grid = [[int(data[i * 3 + j]) for j in range(3)] for i in range(3)]

    total_permutations = 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2
    valid_permutations = 0

    for perm in itertools.permutations(range(9)):
        if not is_disappointed(grid, perm):
            valid_permutations += 1

    probability = valid_permutations / total_permutations
    print(f"{probability:.20f}")

if __name__ == "__main__":
    main()