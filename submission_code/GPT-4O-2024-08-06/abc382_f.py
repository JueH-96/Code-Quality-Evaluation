# YOUR CODE HERE
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
        C = int(data[index + 1])
        L = int(data[index + 2])
        bars.append((R, C, L))
        index += 3
    
    # Create a grid to track the final positions of bars
    grid = [[0] * (W + 1) for _ in range(H + 1)]
    
    # Result array to store the final row of each bar
    final_rows = [0] * N
    
    # Process each bar from the last to the first
    for i in range(N):
        R, C, L = bars[i]
        
        # Determine the lowest row this bar can fall to
        current_row = R
        while current_row < H:
            can_fall = True
            for j in range(L):
                if grid[current_row + 1][C + j] != 0:
                    can_fall = False
                    break
            if not can_fall:
                break
            current_row += 1
        
        # Mark the final position of this bar in the grid
        for j in range(L):
            grid[current_row][C + j] = i + 1
        
        # Store the final row for this bar
        final_rows[i] = current_row
    
    # Output the final rows for each bar
    for row in final_rows:
        print(row)

if __name__ == "__main__":
    main()