import sys

# Read all input and split into a list
data = sys.stdin.read().split()
index = 0

# Read H, W, D
H = int(data[index])
index += 1
W = int(data[index])
index += 1
D = int(data[index])
index += 1

# Read the grid
grid = []
for _ in range(H):
    row_str = data[index]
    grid.append(row_str)
    index += 1

# Find all floor positions (0-based indices)
floors = []
for r in range(H):
    for c in range(W):
        if grid[r][c] == '.':
            floors.append((r, c))

num_floors = len(floors)

# Initialize maximum count of humidified floor cells
max_count = 0

# Iterate over all unordered pairs of distinct floor cells
for i in range(num_floors):
    for j in range(i + 1, num_floors):
        # Positions of the two humidifiers
        hum1_r, hum1_c = floors[i]
        hum2_r, hum2_c = floors[j]
        
        # Count the number of humidified floor cells for this pair
        count = 0
        for k in range(num_floors):
            target_r, target_c = floors[k]
            dist1 = abs(target_r - hum1_r) + abs(target_c - hum1_c)
            dist2 = abs(target_r - hum2_r) + abs(target_c - hum2_c)
            if dist1 <= D or dist2 <= D:
                count += 1
        
        # Update maximum count
        if count > max_count:
            max_count = count

# Output the maximum count
print(max_count)