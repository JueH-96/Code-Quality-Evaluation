def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    H = list(map(int, data[1:N+1]))
    
    # Group positions by height
    height_positions = {}
    for pos in range(1, N+1):
        height = H[pos-1]
        if height not in height_positions:
            height_positions[height] = []
        height_positions[height].append(pos)
    
    max_length = 1  # At least one building can be chosen
    
    for height in height_positions:
        positions = height_positions[height]
        k = len(positions)
        if k == 1:
            continue  # Only one building, which is already considered
        
        positions_set = set(positions)
        current_max = 1
        
        # Iterate through all possible differences d
        for d in range(1, positions[-1] - positions[0] + 1):
            count = 1
            current = positions[0]
            while current + d in positions_set:
                current += d
                count += 1
            if count > current_max:
                current_max = count
        if current_max > max_length:
            max_length = current_max
    
    print(max_length)

if __name__ == "__main__":
    main()