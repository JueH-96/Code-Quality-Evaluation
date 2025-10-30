def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    
    W = int(data[index])
    index += 1
    H = int(data[index])
    index += 1
    N = int(data[index])
    index += 1
    
    p = []
    q = []
    for _ in range(N):
        p.append(int(data[index]))
        index += 1
        q.append(int(data[index]))
        index += 1
    
    A = int(data[index])
    index += 1
    a = []
    for _ in range(A):
        a.append(int(data[index]))
        index += 1
    
    B = int(data[index])
    index += 1
    b = []
    for _ in range(B):
        b.append(int(data[index]))
        index += 1
    
    # Step 1: Coordinate compression for x
    x_positions = a + [0, W] + p
    x_sorted = sorted(set(x_positions))
    x_dict = {x: i for i, x in enumerate(x_sorted)}
    # Step 2: Coordinate compression for y
    y_positions = b + [0, H] + q
    y_sorted = sorted(set(y_positions))
    y_dict = {y: i for i, y in enumerate(y_sorted)}
    
    # Sort strawberries by compressed x and y
    strawberries = sorted([(x_dict[p[i]], y_dict[q[i]]) for i in range(N)], key=lambda x: (x[0], x[1]))
    
    # Sort x_cuts and y_cuts
    x_cuts = sorted(x_positions)
    y_cuts = sorted(y_positions)
    
    # Determine m
    total_pieces = (A + 1) * (B + 1)
    if N < total_pieces:
        m = 0
    else:
        m = 1
    
    # Determine M
    M = 0
    # Iterate through x-segments
    for i in range(len(x_cuts)-1):
        x_start = x_cuts[i]
        x_end = x_cuts[i+1]
        # Find all strawberries in this x-segment
        left = 0
        right = N
        # Find the first strawberry with x >= x_start
        while left < right:
            mid = (left + right) // 2
            if strawberries[mid][0] >= x_dict[x_start]:
                right = mid
            else:
                left = mid + 1
        start_idx = left
        # Find the first strawberry with x >= x_end
        left = 0
        right = N
        while left < right:
            mid = (left + right) // 2
            if strawberries[mid][0] >= x_dict[x_end]:
                right = mid
            else:
                left = mid + 1
        end_idx = left
        # Extract y-coordinates in this x-segment
        y_coords = [strawberries[k][1] for k in range(start_idx, end_idx)]
        # Sort y-coordinates
        y_coords_sorted = sorted(y_coords)
        # Iterate through y-segments and find the one with the most strawberries
        # Using sliding window over y_cuts
        max_in_x = 0
        ptr = 0
        for j in range(len(y_cuts)-1):
            y_start = y_cuts[j]
            y_end = y_cuts[j+1]
            # Count strawberries with y in [y_start, y_end)
            # Find the first y >= y_start
            while ptr < len(y_coords_sorted) and y_coords_sorted[ptr] < y_dict[y_start]:
                ptr += 1
            # Find the first y >= y_end
            left_count = ptr
            while ptr < len(y_coords_sorted) and y_coords_sorted[ptr] < y_dict[y_end]:
                ptr += 1
            right_count = ptr
            count = right_count - left_count
            if count > max_in_x:
                max_in_x = count
        if max_in_x > M:
            M = max_in_x
    print(f"{m} {M}")

if __name__ == "__main__":
    main()