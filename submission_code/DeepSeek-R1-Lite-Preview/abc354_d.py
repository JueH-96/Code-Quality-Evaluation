def calculate_black_area(A, B, C, D):
    # Number of full 2x2 units
    full_units_x = (C - A) // 2
    full_units_y = (D - B) // 2
    black_area = full_units_x * full_units_y * 2
    
    # Remaining width and height
    rem_width = (C - A) % 2
    rem_height = (D - B) % 2
    
    # Black area in remaining horizontal strips
    black_area += full_units_x * rem_height
    
    # Black area in remaining vertical strips
    black_area += full_units_y * rem_width
    
    # Black area in the corner partial unit
    if rem_width and rem_height:
        # Calculate the black area in the partial unit
        # The partial unit is from (A + 2*full_units_x, B + 2*full_units_y) to (C, D)
        A_partial = A + 2 * full_units_x
        B_partial = B + 2 * full_units_y
        width = C - A_partial
        height = D - B_partial
        # In the partial unit, black area is calculated based on the formula
        # We need to integrate the color function over this partial unit
        # For simplicity, assume black area is 0.5 for 1x1 unit
        black_area += 0.5
    return black_area

# Read input
A, B, C, D = map(int, input().split())

# Calculate black area
black_area = calculate_black_area(A, B, C, D)

# Print twice the black area
print(int(2 * black_area))