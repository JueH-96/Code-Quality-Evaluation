# YOUR CODE HERE

from itertools import permutations

def is_disappointing(order, grid):
    # Check all rows
    for i in range(3):
        if order.index(grid[i][0]) < order.index(grid[i][1]) < order.index(grid[i][2]):
            if grid[i][0] == grid[i][1] and grid[i][1] != grid[i][2]:
                return True
            if grid[i][0] != grid[i][1] and grid[i][1] == grid[i][2]:
                return True

    # Check all columns
    for j in range(3):
        if order.index(grid[0][j]) < order.index(grid[1][j]) < order.index(grid[2][j]):
            if grid[0][j] == grid[1][j] and grid[1][j] != grid[2][j]:
                return True
            if grid[0][j] != grid[1][j] and grid[1][j] == grid[2][j]:
                return True

    # Check diagonals
    if order.index(grid[0][0]) < order.index(grid[1][1]) < order.index(grid[2][2]):
        if grid[0][0] == grid[1][1] and grid[1][1] != grid[2][2]:
            return True
        if grid[0][0] != grid[1][1] and grid[1][1] == grid[2][2]:
            return True

    if order.index(grid[2][0]) < order.index(grid[1][1]) < order.index(grid[0][2]):
        if grid[2][0] == grid[1][1] and grid[1][1] != grid[0][2]:
            return True
        if grid[2][0] != grid[1][1] and grid[1][1] == grid[0][2]:
            return True

    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    grid = [
        [int(data[0]), int(data[1]), int(data[2])],
        [int(data[3]), int(data[4]), int(data[5])],
        [int(data[6]), int(data[7]), int(data[8])]
    ]
    
    all_numbers = [grid[i][j] for i in range(3) for j in range(3)]
    all_permutations = permutations(all_numbers)
    
    total_permutations = 0
    non_disappointing_permutations = 0
    
    for perm in all_permutations:
        total_permutations += 1
        if not is_disappointing(perm, grid):
            non_disappointing_permutations += 1
    
    probability = non_disappointing_permutations / total_permutations
    print(f"{probability:.25f}")

if __name__ == "__main__":
    main()