import itertools

def read_input():
    grid = []
    for _ in range(3):
        row = list(map(int, input().split()))
        grid.append(row)
    return grid

def is_disappointed(sequence, grid):
    for i in range(7):
        if i < 3:
            if grid[0][i] == grid[0][i+1] and grid[0][i] != grid[0][i+2]:
                return True
            if grid[1][i] == grid[1][i+1] and grid[1][i] != grid[1][i+2]:
                return True
            if grid[2][i] == grid[2][i+1] and grid[2][i] != grid[2][i+2]:
                return True
        elif i < 5:
            if grid[i-3][0] == grid[i-2][0] and grid[i-3][0] != grid[i-1][0]:
                return True
            if grid[i-3][1] == grid[i-2][1] and grid[i-3][1] != grid[i-1][1]:
                return True
            if grid[i-3][2] == grid[i-2][2] and grid[i-3][2] != grid[i-1][2]:
                return True
        elif i == 5:
            if grid[0][0] == grid[1][1] and grid[0][0] != grid[2][2]:
                return True
        else:
            if grid[2][0] == grid[1][1] and grid[2][0] != grid[0][2]:
                return True
    return False

def calculate_probability(grid):
    all_permutations = list(itertools.permutations(range(9)))
    valid_sequences = 0
    for perm in all_permutations:
        sequence = [grid[perm[i] // 3][perm[i] % 3] for i in range(9)]
        if not is_disappointed(sequence, grid):
            valid_sequences += 1
    return valid_sequences / len(all_permutations)

grid = read_input()
probability = calculate_probability(grid)
print(probability)