def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    X = list(map(int, data[2:2+M]))
    A = list(map(int, data[2+M:2+2*M]))
    
    # Combine X and A into a list of tuples and sort by X
    cells = sorted(zip(X, A), key=lambda x: x[0])
    
    # Calculate the total number of stones
    total_stones = sum(A)
    
    # If the total number of stones is not equal to N, it's impossible
    if total_stones != N:
        print(-1)
        return
    
    # Initialize the number of operations
    operations = 0
    # Initialize the current number of stones in the previous cell
    prev_stones = 0
    
    for i in range(M):
        x, a = cells[i]
        # The number of stones that need to be moved from this cell to the next
        # is the excess stones in this cell minus the required stones (1 per cell)
        # The required stones for the cells between the previous cell and this cell
        # is the number of cells between them
        if i == 0:
            # For the first cell, the number of cells before it is x - 1
            # Each of these cells needs 1 stone
            # So the first cell must have at least (x - 1) stones to cover these cells
            required = x - 1
            if a < required:
                print(-1)
                return
            # The excess stones in the first cell is a - required
            excess = a - required
            # These excess stones will be moved to the next cells
            operations += excess
            prev_stones = excess
        else:
            # For the subsequent cells, the number of cells between the previous cell and this cell
            # is x - cells[i-1][0] - 1
            # Each of these cells needs 1 stone
            # So the previous cell must have at least (x - cells[i-1][0] - 1) stones to cover these cells
            required = x - cells[i-1][0] - 1
            if prev_stones < required:
                print(-1)
                return
            # The excess stones in the previous cell is prev_stones - required
            # These excess stones will be moved to this cell
            excess = prev_stones - required
            # The total stones in this cell after receiving the excess is a + excess
            # The required stones for this cell is 1
            # So the excess stones in this cell is (a + excess) - 1
            total = a + excess
            if total < 1:
                print(-1)
                return
            excess_current = total - 1
            # The number of operations is the excess stones moved from the previous cell to this cell
            operations += excess
            prev_stones = excess_current
    
    # After processing all cells, check if the last cell has exactly 1 stone
    if prev_stones != 1:
        print(-1)
        return
    
    print(operations)

if __name__ == "__main__":
    main()