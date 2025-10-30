def min_operations_to_remove_blacks(n, k, s):
    # Count the initial number of black cells
    black_cells = s.count('B')
    
    # If there are no black cells or k is greater than or equal to the number of black cells, no operation is needed
    if black_cells == 0 or k >= black_cells:
        return 0
    
    # Initialize the number of black cells in the current window and the minimum found so far
    current_blacks = min_blacks = s[:k].count('B')
    
    # Slide the window of size k across the strip and find the window with the minimum number of black cells
    for i in range(k, n):
        # If we move out a black cell, decrease the count
        if s[i - k] == 'B':
            current_blacks -= 1
        # If we move in a black cell, increase the count
        if s[i] == 'B':
            current_blacks += 1
        # Update the minimum number of black cells found in a window so far
        min_blacks = min(min_blacks, current_blacks)
    
    # The minimum number of operations is the minimum number of black cells in any window of size k
    return min_blacks

# Read the number of test cases
t = int(input().strip())

# Process each test case
for _ in range(t):
    # Read the values of n and k
    n, k = map(int, input().strip().split())
    # Read the strip of paper
    s = input().strip()
    # Output the minimum number of operations needed to remove all black cells
    print(min_operations_to_remove_blacks(n, k, s))