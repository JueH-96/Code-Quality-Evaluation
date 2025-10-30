import math

def solve():
    A, B, C, D = map(int, input().split())
    
    total_black_area = 0
    
    # Find range of i values
    for i in range(A, C):
        # Find range of j values  
        j_min = math.floor(B / 2)
        j_max = math.floor((D - 1) / 2)
        
        for j in range(j_min, j_max + 1):
            # Find valid k values for this (i,j)
            # We need i + 2j - 2k in {-2, -1, 0, 1}
            # So 2k in {i + 2j + 2, i + 2j + 1, i + 2j, i + 2j - 1}
            # Since k is integer, 2k is even, so 2k in {i + 2j + 2, i + 2j}
            # Thus k in {(i + 2j)/2 + 1, (i + 2j)/2}
            
            for k_offset in [0, 1]:
                k = (i + 2*j) // 2 + k_offset
                
                # Check if this region is black
                if (i + j + k) % 2 == 0:  # black region
                    # Calculate area of intersection with rectangle
                    x_min = max(i, A)
                    x_max = min(i + 1, C)
                    y_min = max(2*j, B)
                    y_max = min(2*j + 2, D)
                    
                    if x_min < x_max and y_min < y_max:
                        # Calculate area within diagonal constraints 2k < x+y < 2k+2
                        area = 0
                        
                        # Integrate over x from x_min to x_max
                        for x_int in range(math.ceil(x_min * 1000), math.floor(x_max * 1000) + 1):
                            x = x_int / 1000.0
                            if x < x_min or x >= x_max:
                                continue
                                
                            # For this x, find valid y range
                            y_lower = max(y_min, 2*k - x)
                            y_upper = min(y_max, 2*k + 2 - x)
                            
                            if y_lower < y_upper:
                                area += (y_upper - y_lower) * 0.001
                        
                        # More precise calculation using integration
                        # Area = integral of (min(y_max, 2k+2-x) - max(y_min, 2k-x)) dx
                        def integrate_region():
                            total = 0
                            dx = (x_max - x_min) / 10000
                            for i in range(10000):
                                x = x_min + i * dx
                                y_lower = max(y_min, 2*k - x)
                                y_upper = min(y_max, 2*k + 2 - x)
                                if y_lower < y_upper:
                                    total += (y_upper - y_lower) * dx
                            return total
                        
                        area = integrate_region()
                        total_black_area += area
    
    # Actually, let me implement this more directly using geometric formulas
    total_black_area = 0
    
    for i in range(A, C):
        j_min = math.floor(B / 2)
        j_max = math.floor((D - 1) / 2)
        
        for j in range(j_min, j_max + 1):
            # Valid k values
            base_k = (i + 2*j) // 2
            for k in [base_k, base_k + 1]:
                if (i + j + k) % 2 == 0:  # black region
                    # Calculate intersection area
                    x_min = max(i, A)
                    x_max = min(i + 1, C)
                    y_min = max(2*j, B)
                    y_max = min(2*j + 2, D)
                    
                    if x_min < x_max and y_min < y_max:
                        # Calculate area within 2k < x+y < 2k+2
                        # This is a polygon intersection problem
                        
                        # Use the shoelace formula approach
                        vertices = []
                        
                        # Check intersections of rectangle edges with diagonal lines
                        # Bottom edge y = y_min with diagonals
                        x1 = 2*k - y_min
                        x2 = 2*k + 2 - y_min
                        if x_min <= x1 <= x_max:
                            vertices.append((x1, y_min))
                        if x_min <= x2 <= x_max:
                            vertices.append((x2, y_min))
                            
                        # Top edge y = y_max with diagonals  
                        x1 = 2*k - y_max
                        x2 = 2*k + 2 - y_max
                        if x_min <= x1 <= x_max:
                            vertices.append((x1, y_max))
                        if x_min <= x2 <= x_max:
                            vertices.append((x2, y_max))
                            
                        # Left edge x = x_min with diagonals
                        y1 = 2*k - x_min
                        y2 = 2*k + 2 - x_min
                        if y_min <= y1 <= y_max:
                            vertices.append((x_min, y1))
                        if y_min <= y2 <= y_max:
                            vertices.append((x_min, y2))
                            
                        # Right edge x = x_max with diagonals
                        y1 = 2*k - x_max
                        y2 = 2*k + 2 - x_max
                        if y_min <= y1 <= y_max:
                            vertices.append((x_max, y1))
                        if y_min <= y2 <= y_max:
                            vertices.append((x_max, y2))
                            
                        # Add rectangle corners that are inside the strip
                        corners = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
                        for x, y in corners:
                            if 2*k < x + y < 2*k + 2:
                                vertices.append((x, y))
                        
                        if len(vertices) >= 3:
                            # Remove duplicates and sort vertices
                            vertices = list(set(vertices))
                            if len(vertices) >= 3:
                                # Calculate area using shoelace formula
                                # First sort vertices in counterclockwise order
                                cx = sum(p[0] for p in vertices) / len(vertices)
                                cy = sum(p[1] for p in vertices) / len(vertices)
                                vertices.sort(key=lambda p: math.atan2(p[1]-cy, p[0]-cx))
                                
                                area = 0
                                n = len(vertices)
                                for i in range(n):
                                    j = (i + 1) % n
                                    area += vertices[i][0] * vertices[j][1]
                                    area -= vertices[j][0] * vertices[i][1]
                                area = abs(area) / 2
                                total_black_area += area
    
    return int(2 * total_black_area + 0.5)

print(solve())