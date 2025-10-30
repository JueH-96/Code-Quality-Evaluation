def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and W
    N, W = map(int, data[0].split())
    
    # Read the positions of the blocks
    blocks = {}
    for i in range(1, N + 1):
        x, y = map(int, data[i].split())
        blocks[i] = (x, y)
    
    # Read number of queries
    Q = int(data[N + 1])
    
    # Prepare to answer queries
    results = []
    
    # For each query
    for i in range(N + 2, N + 2 + Q):
        T, A = map(int, data[i].split())
        x, y = blocks[A]
        
        # Calculate the maximum height of the blocks in column x
        max_height = 0
        for j in range(1, N + 1):
            bx, by = blocks[j]
            if bx == x:
                max_height = max(max_height, by)
        
        # Determine the time when the block A would be removed
        if max_height >= 1 and max_height <= T:
            # The block A will be removed if the bottom row is filled
            if T >= 1 and T >= (max_height - 1):
                results.append("No")
            else:
                results.append("Yes")
        else:
            # If the block is above the maximum height that can be filled
            if T >= max_height:
                results.append("No")
            else:
                results.append("Yes")
    
    # Print results
    sys.stdout.write("
".join(results) + "
")

if __name__ == "__main__":
    main()