def main():
    import sys
    from bisect import bisect_left, bisect_right

    input_data = sys.stdin.read().strip().split()
    # Parse initial input
    N = int(input_data[0])
    M = int(input_data[1])
    Sx = int(input_data[2])
    Sy = int(input_data[3])

    # Read houses
    # We'll store:
    #   dict_x: x -> sorted list of y for houses at x
    #   dict_y: y -> sorted list of x for houses at y
    #   house_idx: (x,y) -> index in [0..N-1]
    dict_x = {}
    dict_y = {}
    house_idx = {}

    pos = 4
    for i in range(N):
        x = int(input_data[pos]); y = int(input_data[pos+1])
        pos += 2
        if x not in dict_x:
            dict_x[x] = []
        dict_x[x].append(y)

        if y not in dict_y:
            dict_y[y] = []
        dict_y[y].append(x)

        house_idx[(x, y)] = i

    # Sort the lists
    for x in dict_x:
        dict_x[x].sort()
    for y in dict_y:
        dict_y[y].sort()

    # Commands
    commands = []
    for _ in range(M):
        d = input_data[pos]
        c = int(input_data[pos+1])
        pos += 2
        commands.append((d, c))

    # Visited array
    visited = [False]*N
    visited_count = 0

    # Current position
    curx, cury = Sx, Sy

    # Define removal functions
    def remove_vertical(x, lowY, highY):
        # Removes all houses on vertical line x in the interval [lowY..highY]
        # Marks them visited if not already, and removes from dict_y as well
        nonlocal visited_count
        if x not in dict_x:
            return
        arr = dict_x[x]
        start = bisect_left(arr, lowY)
        end   = bisect_right(arr, highY)
        segment = arr[start:end]

        for y in segment:
            # Check if this (x,y) is truly a house
            if (x,y) in house_idx:
                hidx = house_idx[(x,y)]
                if not visited[hidx]:
                    visited[hidx] = True
                    visited_count += 1
                # Remove x from dict_y[y]
                row = dict_y[y]
                ix = bisect_left(row, x)
                if ix < len(row) and row[ix] == x:
                    row.pop(ix)
        # Now remove them from dict_x[x]
        arr[start:end] = []

    def remove_horizontal(y, lowX, highX):
        # Removes all houses on horizontal line y in the interval [lowX..highX]
        # Marks them visited if not already, and removes from dict_x as well
        nonlocal visited_count
        if y not in dict_y:
            return
        arr = dict_y[y]
        start = bisect_left(arr, lowX)
        end   = bisect_right(arr, highX)
        segment = arr[start:end]

        for x in segment:
            # Check if this (x,y) is truly a house
            if (x,y) in house_idx:
                hidx = house_idx[(x,y)]
                if not visited[hidx]:
                    visited[hidx] = True
                    visited_count += 1
                # Remove y from dict_x[x]
                row = dict_x[x]
                iy = bisect_left(row, y)
                if iy < len(row) and row[iy] == y:
                    row.pop(iy)
        # Now remove them from dict_y[y]
        arr[start:end] = []

    # Process commands
    for (d, c) in commands:
        if d == 'U':
            newy = cury + c
            lowY, highY = min(cury, newy), max(cury, newy)
            remove_vertical(curx, lowY, highY)
            cury = newy
        elif d == 'D':
            newy = cury - c
            lowY, highY = min(cury, newy), max(cury, newy)
            remove_vertical(curx, lowY, highY)
            cury = newy
        elif d == 'L':
            newx = curx - c
            lowX, highX = min(curx, newx), max(curx, newx)
            remove_horizontal(cury, lowX, highX)
            curx = newx
        else:  # d == 'R'
            newx = curx + c
            lowX, highX = min(curx, newx), max(curx, newx)
            remove_horizontal(cury, lowX, highX)
            curx = newx

    # Output final position and number of distinct visited houses
    print(curx, cury, visited_count)

# Do not forget to call main
if __name__ == "__main__":
    main()