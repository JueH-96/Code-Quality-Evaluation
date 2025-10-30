# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    grid = [list(row) for row in data[1:N+1]]
    
    for i in range(1, N//2 + 1):
        for x in range(i, N + 1 - i):
            for y in range(i, N + 1 - i):
                # Calculate the target cell
                target_x = y
                target_y = N + 1 - x
                # Store the original color
                original_color = grid[x-1][y-1]
                # Update the target cell
                grid[target_x-1][target_y-1] = original_color
    
    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    main()