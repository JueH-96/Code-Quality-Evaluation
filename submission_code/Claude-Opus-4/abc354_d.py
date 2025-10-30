# YOUR CODE HERE
def solve(A, B, C, D):
    # The key insight is that the pattern has a specific structure
    # We need to calculate the black area in the rectangle
    
    # Helper function to calculate black area from (0,0) to (x,y)
    def black_area_from_origin(x, y):
        if x <= 0 or y <= 0:
            return 0
        
        # The pattern repeats every 2 units in both x and y
        # In each 2x2 square, exactly 2 units are black
        
        # Number of complete 2x2 squares
        complete_x = x // 2
        complete_y = y // 2
        area = complete_x * complete_y * 2
        
        # Remaining strip on the right
        if x % 2 == 1:
            area += complete_y
            if y % 2 == 1:
                area += 0.5
        
        # Remaining strip on top
        if y % 2 == 1:
            area += complete_x
        
        return area
    
    # Use inclusion-exclusion principle
    # Area in rectangle = Area(C,D) - Area(A,D) - Area(C,B) + Area(A,B)
    
    # Adjust coordinates to handle negative values
    # The pattern is symmetric, so we can shift everything
    offset_x = 0
    offset_y = 0
    
    if A < 0:
        offset_x = ((-A + 1) // 2) * 2
    if B < 0:
        offset_y = ((-B + 1) // 2) * 2
    
    A += offset_x
    B += offset_y
    C += offset_x
    D += offset_y
    
    # Calculate using inclusion-exclusion
    area = black_area_from_origin(C, D) - black_area_from_origin(A, D) - black_area_from_origin(C, B) + black_area_from_origin(A, B)
    
    # Adjust for the shift based on parity
    if offset_x % 4 == 2:
        area = (C - A) * (D - B) - area
    if offset_y % 4 == 2:
        area = (C - A) * (D - B) - area
    
    return int(2 * area)

# Read input
A, B, C, D = map(int, input().split())

# Solve and print result
print(solve(A, B, C, D))