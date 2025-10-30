def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    if M == 0:
        print("Yes")
        return
    
    # Parse the pre-colored cells
    precolored = []
    index = 2
    for _ in range(M):
        X = int(data[index]) - 1
        Y = int(data[index+1]) - 1
        C = data[index+2]
        precolored.append((X, Y, C))
        index += 3
    
    # To satisfy the conditions, we need to check the constraints on rows and columns
    # We use two dictionaries to track the required black and white boundaries
    row_black_max = {}
    row_white_min = {}
    col_black_max = {}
    col_white_min = {}
    
    for x, y, c in precolored:
        if c == 'B':
            if x in row_black_max:
                row_black_max[x] = max(row_black_max[x], y)
            else:
                row_black_max[x] = y
            
            if y in col_black_max:
                col_black_max[y] = max(col_black_max[y], x)
            else:
                col_black_max[y] = x
        elif c == 'W':
            if x in row_white_min:
                row_white_min[x] = min(row_white_min[x], y)
            else:
                row_white_min[x] = y
            
            if y in col_white_min:
                col_white_min[y] = min(col_white_min[y], x)
            else:
                col_white_min[y] = x
    
    # Now validate the constraints
    for x in row_black_max:
        if x in row_white_min:
            if row_black_max[x] >= row_white_min[x]:
                print("No")
                return
    
    for y in col_black_max:
        if y in col_white_min:
            if col_black_max[y] >= col_white_min[y]:
                print("No")
                return
    
    # If all checks are passed, it's possible to color the grid
    print("Yes")