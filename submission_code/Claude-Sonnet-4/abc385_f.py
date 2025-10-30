n = int(input())
buildings = []
for _ in range(n):
    x, h = map(int, input().split())
    buildings.append((x, h))

max_blocked_height = -1

for i in range(n):
    x_i, h_i = buildings[i]
    min_height_to_see = float('inf')
    
    for j in range(i):
        x_j, h_j = buildings[j]
        # Calculate the minimum height needed to see building i over building j
        # h < (h_j * x_i - h_i * x_j) / (x_i - x_j)
        if x_i > x_j:  # This should always be true given the input constraints
            threshold = (h_j * x_i - h_i * x_j) / (x_i - x_j)
            if threshold > 0:
                min_height_to_see = min(min_height_to_see, threshold)
    
    if min_height_to_see != float('inf'):
        max_blocked_height = max(max_blocked_height, min_height_to_see)

if max_blocked_height == -1:
    print(-1)
else:
    print(f"{max_blocked_height:.18f}")