import bisect

def main():
    # Read input
    N, M, S_x, S_y = map(int, input().split())
    
    # Group houses by x and y coordinates
    houses_with_x = {}
    houses_with_y = {}
    
    for _ in range(N):
        X, Y = map(int, input().split())
        if X not in houses_with_x:
            houses_with_x[X] = []
        houses_with_x[X].append(Y)
        
        if Y not in houses_with_y:
            houses_with_y[Y] = []
        houses_with_y[Y].append(X)
    
    # Sort the lists for binary search
    for x in houses_with_x:
        houses_with_x[x].sort()
    for y in houses_with_y:
        houses_with_y[y].sort()
    
    directions = []
    for _ in range(M):
        D, C = input().split()
        C = int(C)
        directions.append((D, C))
    
    # Track Santa's position
    pos_x, pos_y = S_x, S_y
    
    # Track houses he passes through
    passed_houses = set()
    
    for D, C in directions:
        new_x, new_y = pos_x, pos_y
        
        if D == 'U':
            new_y += C
            # Check houses with the same x-coordinate
            if pos_x in houses_with_x:
                ys = houses_with_x[pos_x]
                left = bisect.bisect_left(ys, pos_y)
                right = bisect.bisect_right(ys, new_y)
                for i in range(left, right):
                    passed_houses.add((pos_x, ys[i]))
        elif D == 'D':
            new_y -= C
            # Check houses with the same x-coordinate
            if pos_x in houses_with_x:
                ys = houses_with_x[pos_x]
                left = bisect.bisect_left(ys, new_y)
                right = bisect.bisect_right(ys, pos_y)
                for i in range(left, right):
                    passed_houses.add((pos_x, ys[i]))
        elif D == 'L':
            new_x -= C
            # Check houses with the same y-coordinate
            if pos_y in houses_with_y:
                xs = houses_with_y[pos_y]
                left = bisect.bisect_left(xs, new_x)
                right = bisect.bisect_right(xs, pos_x)
                for i in range(left, right):
                    passed_houses.add((xs[i], pos_y))
        elif D == 'R':
            new_x += C
            # Check houses with the same y-coordinate
            if pos_y in houses_with_y:
                xs = houses_with_y[pos_y]
                left = bisect.bisect_left(xs, pos_x)
                right = bisect.bisect_right(xs, new_x)
                for i in range(left, right):
                    passed_houses.add((xs[i], pos_y))
        
        pos_x, pos_y = new_x, new_y
    
    print(pos_x, pos_y, len(passed_houses))

if __name__ == "__main__":
    main()