def calculate_black_area(A, B, C, D):
    import math
    
    # Floor values of A, B, C, D
    floorA = math.floor(A)
    floorB = math.floor(B)
    floorC = math.floor(C)
    floorD = math.floor(D)
    
    # Number of complete x-intervals
    num_complete_x = floorC - floorA
    # Number of complete y-intervals of height 2
    num_complete_y = (floorD // 2) - (floorB // 2)
    
    # Black area from complete fundamental regions
    black_complete = 0
    for k in range(num_complete_x):
        if (floorA + k) % 2 == 0:
            black_complete += 1.5 * num_complete_y
        else:
            black_complete += 0.5 * num_complete_y
    
    # Partial y-interval at the top
    partial_y_height = D - (floorD // 2 * 2)
    if partial_y_height > 0:
        for k in range(num_complete_x):
            x_start = floorA + k
            if x_start % 2 == 0:
                black_partial = (1.5 / 2) * partial_y_height
            else:
                black_partial = (0.5 / 2) * partial_y_height
            black_complete += black_partial
    
    # Handle partial x-intervals on the left and right
    partial_x_left = floorA + 1 - A if floorA >= A else 1
    partial_x_right = C - floorC if floorC < C else 0
    
    # Add partial x-intervals to the black area
    for m in range((floorB // 2), (floorD // 2) + 1):
        y_start = 2 * m
        for k in range(floorA, floorC):
            x_start = k
            if k % 2 == 0:
                black_area = 1.5
            else:
                black_area = 0.5
            # Adjust for partial x-intervals
            if k == floorA:
                black_area *= partial_x_left
            if k == floorC - 1:
                black_area *= partial_x_right
            black_complete += black_area
    
    # Handle corners if necessary
    # ...
    
    # Twice the black area
    result = 2 * black_complete
    return int(result)

# Read input
A, B, C, D = map(int, input().split())

# Calculate and print the result
print(calculate_black_area(A, B, C, D))