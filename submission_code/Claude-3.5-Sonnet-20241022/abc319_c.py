from itertools import permutations
import sys

def check_disappointment(order, grid):
    # Check all lines for disappointment
    lines = [
        [(0,0), (0,1), (0,2)], # row 1
        [(1,0), (1,1), (1,2)], # row 2 
        [(2,0), (2,1), (2,2)], # row 3
        [(0,0), (1,0), (2,0)], # col 1
        [(0,1), (1,1), (2,1)], # col 2
        [(0,2), (1,2), (2,2)], # col 3
        [(0,0), (1,1), (2,2)], # diagonal 1
        [(2,0), (1,1), (0,2)]  # diagonal 2
    ]
    
    seen_pos = set()
    for pos in order:
        seen_pos.add(pos)
        
        # Check each line
        for line in lines:
            seen_in_line = [p for p in line if p in seen_pos]
            if len(seen_in_line) == 2:
                # If first two seen numbers in line are same but third is different
                vals = [grid[r][c] for r,c in seen_in_line]
                if vals[0] == vals[1]:
                    last_pos = [p for p in line if p not in seen_pos][0]
                    if grid[last_pos[0]][last_pos[1]] != vals[0]:
                        return False
    return True

# Read input
grid = []
for _ in range(3):
    grid.append(list(map(int, input().split())))

# Generate all possible orders
positions = [(i,j) for i in range(3) for j in range(3)]
total = 0
valid = 0

for perm in permutations(positions):
    total += 1
    if check_disappointment(perm, grid):
        valid += 1

print(valid/total)