H, W = map(int, input().split())
grid = []
for _ in range(H):
    row = input().strip()
    grid.append([int(c) for c in row])

# Precompute row masks
row_mask = []
for i in range(H):
    mask = 0
    for j in range(W):
        if grid[i][j] == 1:
            mask |= (1 << j)
    row_mask.append(mask)

# Precompute popcount
popcount = [0] * (1 << W)
for i in range(1 << W):
    popcount[i] = bin(i).count('1')

min_sum = float('inf')

# Enumerate all possible choices of which columns to flip
for y_mask in range(1 << W):
    total_sum = 0
    for i in range(H):
        # Compute the XOR of row i and y
        xor_result = row_mask[i] ^ y_mask
        # Count the number of 1s
        count1 = popcount[xor_result]
        count0 = W - count1
        
        # The optimal contribution is the minimum of flipping or not flipping
        contribution = min(count0, count1)
        total_sum += contribution
    
    min_sum = min(min_sum, total_sum)

print(min_sum)