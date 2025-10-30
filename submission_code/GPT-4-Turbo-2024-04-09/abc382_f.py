def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    bars = []
    index = 3
    for _ in range(N):
        R = int(data[index])
        C = int(data[index+1])
        L = int(data[index+2])
        bars.append((R, C, L))
        index += 3
    
    # To find the final position of each bar, we need to simulate the falling process.
    # We can use a sweep line algorithm to efficiently determine the lowest possible row each bar can reach.
    
    # We will use a segment tree or a similar structure to keep track of the lowest occupied cell in each column.
    # We will process each bar in order of their row positions.
    
    # Sort bars by their starting row R_i
    bars.sort()
    
    # We will use a list to represent the lowest occupied cell in each column.
    # Initially, all columns are unoccupied below the grid, so we set them to H+1 (1-indexed).
    lowest_occupied = [H+1] * (W + 1)
    
    # Process each bar
    results = [0] * N
    for i, (R, C, L) in enumerate(bars):
        # Check if the bar can fall down
        min_lowest = H + 1
        for j in range(C, C + L):
            min_lowest = min(min_lowest, lowest_occupied[j] - 1)
        
        # The final row this bar can occupy is the minimum of the lowest cells it can reach
        final_row = min(min_lowest, H)
        
        # Update the lowest occupied cells for this bar's columns
        for j in range(C, C + L):
            lowest_occupied[j] = min(lowest_occupied[j], final_row)
        
        # Store the result for this bar
        results[i] = final_row
    
    # Output the results
    for res in results:
        print(res)

if __name__ == "__main__":
    main()