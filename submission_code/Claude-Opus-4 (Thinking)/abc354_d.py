def solve():
    A, B, C, D = map(int, input().split())
    
    # Function to calculate area of polygon using shoelace formula
    def polygon_area(vertices):
        if len(vertices) < 3:
            return 0
        area = 0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i][0] * vertices[j][1]
            area -= vertices[j][0] * vertices[i][1]
        return abs(area) / 2
    
    total_black_area = 0
    
    # Iterate through all possible regions
    for i in range(A - 2, C + 2):
        for j in range((B - 2) // 2 - 1, (D + 2) // 2 + 2):
            for k in range((A + B - 4) // 2 - 1, (C + D + 4) // 2 + 2):
                # Region bounded by: i <= x < i+1, 2j <= y < 2j+2, 2k <= x+y < 2k+2
                
                # Check if region is valid
                if i + 2*j >= 2*k + 2 or i + 1 + 2*j + 2 <= 2*k:
                    continue
                
                # Find intersection with rectangle [A,C] × [B,D]
                x_min = max(i, A)
                x_max = min(i + 1, C)
                y_min = max(2 * j, B)
                y_max = min(2 * j + 2, D)
                
                if x_min >= x_max or y_min >= y_max:
                    continue
                
                # Find vertices of the intersection polygon
                vertices = []
                
                # Check corner points
                for x in [x_min, x_max]:
                    for y in [y_min, y_max]:
                        if 2*k <= x + y <= 2*k + 2:
                            vertices.append((x, y))
                
                # Check intersections with diagonal boundaries
                # x + y = 2k
                if 2*k >= x_min + y_min and 2*k <= x_max + y_max:
                    # Left edge
                    y = 2*k - x_min
                    if y_min <= y <= y_max:
                        vertices.append((x_min, y))
                    # Right edge
                    y = 2*k - x_max
                    if y_min <= y <= y_max:
                        vertices.append((x_max, y))
                    # Bottom edge
                    x = 2*k - y_min
                    if x_min <= x <= x_max:
                        vertices.append((x, y_min))
                    # Top edge
                    x = 2*k - y_max
                    if x_min <= x <= x_max:
                        vertices.append((x, y_max))
                
                # x + y = 2k + 2
                if 2*k + 2 >= x_min + y_min and 2*k + 2 <= x_max + y_max:
                    # Left edge
                    y = 2*k + 2 - x_min
                    if y_min <= y <= y_max:
                        vertices.append((x_min, y))
                    # Right edge
                    y = 2*k + 2 - x_max
                    if y_min <= y <= y_max:
                        vertices.append((x_max, y))
                    # Bottom edge
                    x = 2*k + 2 - y_min
                    if x_min <= x <= x_max:
                        vertices.append((x, y_min))
                    # Top edge
                    x = 2*k + 2 - y_max
                    if x_min <= x <= x_max:
                        vertices.append((x, y_max))
                
                # Remove duplicates and sort by angle
                vertices = list(set(vertices))
                if len(vertices) < 3:
                    continue
                
                # Sort vertices to form convex polygon
                cx = sum(v[0] for v in vertices) / len(vertices)
                cy = sum(v[1] for v in vertices) / len(vertices)
                import math
                vertices.sort(key=lambda v: math.atan2(v[1] - cy, v[0] - cx))
                
                # Calculate area
                area = polygon_area(vertices)
                
                # Check if region is black
                if (i + j + k) % 2 == 0:
                    total_black_area += area
    
    # Print twice the black area
    print(int(2 * total_black_area + 0.5))

solve()