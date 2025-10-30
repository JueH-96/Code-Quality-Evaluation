# YOUR CODE HERE
W, H = map(int, input().split())
N = int(input())

strawberries = []
for _ in range(N):
    p, q = map(int, input().split())
    strawberries.append((p, q))

A = int(input())
a_cuts = list(map(int, input().split()))

B = int(input())
b_cuts = list(map(int, input().split()))

# Add boundaries (0 and W for x, 0 and H for y)
x_boundaries = [0] + a_cuts + [W]
y_boundaries = [0] + b_cuts + [H]

# Sort boundaries
x_boundaries.sort()
y_boundaries.sort()

# Count strawberries in each piece
piece_counts = []

# For each piece defined by consecutive x and y boundaries
for i in range(len(x_boundaries) - 1):
    for j in range(len(y_boundaries) - 1):
        x_left = x_boundaries[i]
        x_right = x_boundaries[i + 1]
        y_bottom = y_boundaries[j]
        y_top = y_boundaries[j + 1]
        
        # Count strawberries in this piece
        count = 0
        for p, q in strawberries:
            if x_left < p < x_right and y_bottom < q < y_top:
                count += 1
        
        piece_counts.append(count)

# Find minimum and maximum
min_count = min(piece_counts)
max_count = max(piece_counts)

print(min_count, max_count)