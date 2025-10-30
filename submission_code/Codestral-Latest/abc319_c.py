import sys
from itertools import permutations

def is_valid_ordering(grid, order):
    for i in range(3):
        if grid[order[i][0]][order[i][1]] == grid[order[i+1][0]][order[i+1][1]] == grid[order[i+2][0]][order[i+2][1]]:
            return False
    return True

def count_valid_orderings(grid):
    valid_count = 0
    total_count = 0
    positions = [(i, j) for i in range(3) for j in range(3)]
    for order in permutations(positions):
        total_count += 1
        if is_valid_ordering(grid, order):
            valid_count += 1
    return valid_count, total_count

def main():
    grid = []
    for _ in range(3):
        grid.append(list(map(int, sys.stdin.readline().split())))

    valid_count, total_count = count_valid_orderings(grid)
    probability = valid_count / total_count
    print(probability)

if __name__ == "__main__":
    main()