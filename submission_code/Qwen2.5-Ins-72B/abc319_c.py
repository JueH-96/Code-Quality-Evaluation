import itertools
import sys

def is_disappointing(order, grid):
    seen = [grid[r-1][c-1] for r, c in order]
    for i in range(1, 8):
        if seen[i] == seen[i-1] and seen[i] != seen[i+1]:
            return True
    return False

def probability(grid):
    total_permutations = 362880  # 9!
    valid_permutations = 0

    for perm in itertools.permutations(range(1, 10)):
        order = [(r, c) for r in range(1, 4) for c in range(1, 4)]
        if not is_disappointing(order, grid):
            valid_permutations += 1

    return valid_permutations / total_permutations

# Read input
grid = [list(map(int, input().split())) for _ in range(3)]

# Calculate and print the probability
print(probability(grid))