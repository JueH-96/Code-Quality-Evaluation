def solve():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read H, W, N
    H, W, N = map(int, data[0].split())
    
    # Read bars information
    bars = []
    for i in range(1, N + 1):
        R, C, L = map(int, data[i].split())
        bars.append((R, C, L))
    
    # To keep track of the lowest occupied row for each column
    lowest_row = [H + 1] * (W + 1)  # Using 1-based index for columns
    
    # Initialize the lowest row for each bar
    for R, C, L in bars:
        for j in range(C, C + L):
            lowest_row[j] = min(lowest_row[j], R)
    
    # Result array to store the final positions
    result = [0] * N
    
    # Process each bar
    for i in range(N):
        R, C, L = bars[i]
        
        # Calculate the maximum possible drop for this bar
        max_drop = H - R  # Maximum drop possible is to the bottom row
        for j in range(C, C + L):
            # The bar can only drop to the lowest row that is not occupied
            max_drop = min(max_drop, lowest_row[j] - R - 1)
        
        # The final position of the bar
        result[i] = R + max_drop
        
        # Update the lowest_row for the columns occupied by this bar
        for j in range(C, C + L):
            lowest_row[j] = result[i] + 1  # Update to the row just below the bar
    
    # Print the results
    sys.stdout.write('
'.join(map(str, result)) + '
')