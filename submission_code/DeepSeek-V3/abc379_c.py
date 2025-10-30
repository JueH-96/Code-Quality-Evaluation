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
    
    for i in range(len(cells)):
        x, a = cells[i]
        # The number of stones that need to be moved to the right
        # is the excess stones in the current cell
        excess = a - 1
        if excess > 0:
            # The next cell is x+1, but we need to find the next cell with stones
            # Since we are processing in order, the next cell is the next in the list
            if i < len(cells) - 1:
                next_x = cells[i+1][0]
                # The distance between current cell and next cell
                distance = next_x - x
                # The number of operations is excess * distance
                operations += excess * distance
            else:
                # If it's the last cell, the excess must be zero
                if excess != 0:
                    print(-1)
                    return
        # Update the previous stones
        prev_stones = a
    
    print(operations)

if __name__ == "__main__":
    main()