n = int(input())
rectangles = []
for _ in range(n):
    a, b, c, d = map(int, input().split())
    rectangles.append((a, b, c, d))

# Collect all x and y coordinates
x_coords = set()
y_coords = set()
for a, b, c, d in rectangles:
    x_coords.add(a)
    x_coords.add(b)
    y_coords.add(c)
    y_coords.add(d)

# Sort coordinates
x_coords = sorted(x_coords)
y_coords = sorted(y_coords)

# Calculate total area
total_area = 0
for i in range(len(x_coords) - 1):
    for j in range(len(y_coords) - 1):
        x1, x2 = x_coords[i], x_coords[i + 1]
        y1, y2 = y_coords[j], y_coords[j + 1]
        
        # Check if this cell is covered by any rectangle
        covered = False
        for a, b, c, d in rectangles:
            if a <= x1 and x2 <= b and c <= y1 and y2 <= d:
                covered = True
                break
        
        if covered:
            total_area += (x2 - x1) * (y2 - y1)

print(total_area)