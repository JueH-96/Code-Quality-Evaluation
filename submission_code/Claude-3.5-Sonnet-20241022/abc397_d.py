def find_cube_diff_pair(N):
    # For each y starting from 1
    # x^3 - y^3 = N
    # x^3 = N + y^3
    # x = ∛(N + y^3)
    y = 1
    # We can limit our search because x^3 - y^3 = (x-y)(x^2 + xy + y^2)
    # Since x > y, x-y ≥ 1, and x^2 + xy + y^2 > 3y^2
    # Therefore N = x^3 - y^3 > 3y^3
    # So y < ∛(N/3)
    max_y = int(pow(N/3, 1/3)) + 2  # Add 2 for safety due to floating point precision
    
    while y <= max_y:
        # Calculate x^3 = N + y^3
        y_cube = y * y * y
        x_cube = N + y_cube
        
        # Find if x_cube is a perfect cube
        # Using binary search to find cube root
        left = y  # x must be greater than y
        right = int(pow(x_cube, 1/3)) + 2  # Add 2 for safety
        
        while left <= right:
            mid = (left + right) // 2
            curr_cube = mid * mid * mid
            
            if curr_cube == x_cube:
                return (mid, y)
            elif curr_cube < x_cube:
                left = mid + 1
            else:
                right = mid - 1
        
        y += 1
    
    return (-1, -1)

# Read input
N = int(input())

# Find solution
x, y = find_cube_diff_pair(N)

# Print output
if x == -1:
    print(-1)
else:
    print(f"{x} {y}")