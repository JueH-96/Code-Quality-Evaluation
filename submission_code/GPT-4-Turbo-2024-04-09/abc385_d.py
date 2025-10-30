def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # Read the first line of inputs
    index = 0
    N = int(data[index])
    M = int(data[index + 1])
    S_x = int(data[index + 2])
    S_y = int(data[index + 3])
    index += 4
    
    # Read the house coordinates
    houses = set()
    for _ in range(N):
        X = int(data[index])
        Y = int(data[index + 1])
        index += 2
        houses.add((X, Y))
    
    # Read the movements
    movements = []
    for _ in range(M):
        D = data[index]
        C = int(data[index + 1])
        index += 2
        movements.append((D, C))
    
    # Initial position of Santa
    x, y = S_x, S_y
    
    # Set to track visited houses
    visited_houses = set()
    
    # Process each movement
    for D, C in movements:
        if D == 'U':
            # Moving up from (x, y) to (x, y+C)
            for step in range(1, C + 1):
                if (x, y + step) in houses:
                    visited_houses.add((x, y + step))
            y += C
        elif D == 'D':
            # Moving down from (x, y) to (x, y-C)
            for step in range(1, C + 1):
                if (x, y - step) in houses:
                    visited_houses.add((x, y - step))
            y -= C
        elif D == 'L':
            # Moving left from (x, y) to (x-C, y)
            for step in range(1, C + 1):
                if (x - step, y) in houses:
                    visited_houses.add((x - step, y))
            x -= C
        elif D == 'R':
            # Moving right from (x, y) to (x+C, y)
            for step in range(1, C + 1):
                if (x + step, y) in houses:
                    visited_houses.add((x + step, y))
            x += C
    
    # Output the final position and the count of distinct visited houses
    print(x, y, len(visited_houses))

if __name__ == "__main__":
    main()