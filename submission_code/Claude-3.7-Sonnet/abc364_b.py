# Read the input
H, W = map(int, input().split())
S_i, S_j = map(int, input().split())
grid = []
for _ in range(H):
    grid.append(input())
X = input()

# Initialize Takahashi's position
current_row, current_col = S_i, S_j

# Process each instruction
for instruction in X:
    next_row, next_col = current_row, current_col
    
    if instruction == 'L':
        next_col -= 1
    elif instruction == 'R':
        next_col += 1
    elif instruction == 'U':
        next_row -= 1
    elif instruction == 'D':
        next_row += 1
    
    # Check if the next position is valid
    if 1 <= next_row <= H and 1 <= next_col <= W and grid[next_row-1][next_col-1] == '.':
        current_row, current_col = next_row, next_col

# Print the final position
print(current_row, current_col)