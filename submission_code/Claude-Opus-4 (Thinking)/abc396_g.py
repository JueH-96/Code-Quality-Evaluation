H, W = map(int, input().split())
grid = []
for _ in range(H):
    row = input().strip()
    grid.append([int(ch) for ch in row])

min_sum = float('inf')

# Try all 2^W possibilities for column flips
for col_mask in range(1 << W):
    total_sum = 0
    for i in range(H):
        row_sum = 0
        for j in range(W):
            val = grid[i][j]
            if col_mask & (1 << j):
                val = 1 - val
            row_sum += val
        
        # Choose whether to flip this row
        total_sum += min(row_sum, W - row_sum)
    
    min_sum = min(min_sum, total_sum)

print(min_sum)