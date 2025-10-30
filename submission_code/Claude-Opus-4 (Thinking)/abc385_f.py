# YOUR CODE HERE
n = int(input())
buildings = []
for _ in range(n):
    x, h = map(int, input().split())
    buildings.append((x, h))

max_min_h = float('-inf')

for i in range(1, n):  # Start from second building
    x_i, h_i = buildings[i]
    min_h_i = float('-inf')
    
    for j in range(i):  # Check all previous buildings
        x_j, h_j = buildings[j]
        # Calculate the minimum height needed to see building i over building j
        h_threshold = (h_j * x_i - h_i * x_j) / (x_i - x_j)
        min_h_i = max(min_h_i, h_threshold)
    
    max_min_h = max(max_min_h, min_h_i)

if max_min_h < 0:
    print(-1)
else:
    print(f"{max_min_h:.18f}")