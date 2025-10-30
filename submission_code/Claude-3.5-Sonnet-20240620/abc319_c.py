# YOUR CODE HERE
from itertools import permutations
from fractions import Fraction

def is_disappointed(order, grid):
    lines = [
        [(0,0), (0,1), (0,2)], [(1,0), (1,1), (1,2)], [(2,0), (2,1), (2,2)],
        [(0,0), (1,0), (2,0)], [(0,1), (1,1), (2,1)], [(0,2), (1,2), (2,2)],
        [(0,0), (1,1), (2,2)], [(0,2), (1,1), (2,0)]
    ]
    
    for line in lines:
        seen = set()
        for cell in order:
            if cell in line:
                seen.add(cell)
                if len(seen) == 2:
                    first_two = [grid[i][j] for i, j in seen]
                    if first_two[0] == first_two[1]:
                        last_cell = next(c for c in line if c not in seen)
                        if grid[last_cell[0]][last_cell[1]] != first_two[0]:
                            return True
                    break
    return False

# Read input
grid = [list(map(int, input().split())) for _ in range(3)]

# Generate all possible orders
all_orders = list(permutations([(i, j) for i in range(3) for j in range(3)]))

# Count non-disappointing orders
non_disappointing = sum(1 for order in all_orders if not is_disappointed(order, grid))

# Calculate probability
probability = Fraction(non_disappointing, len(all_orders))

# Print result
print(float(probability))