def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read the first line of input
    first_line = data[0].split()
    N = int(first_line[0])
    M = int(first_line[1])
    S_x = int(first_line[2])
    S_y = int(first_line[3])
    
    # Read the houses' coordinates
    houses = set()
    for i in range(1, N + 1):
        x, y = map(int, data[i].split())
        houses.add((x, y))
    
    # Read the movements
    movements = []
    for i in range(N + 1, N + M + 1):
        direction, distance = data[i].split()
        movements.append((direction, int(distance)))
    
    # Current position of Santa
    current_x = S_x
    current_y = S_y
    
    # Set to track distinct houses visited
    visited_houses = set()
    
    # Process each movement
    for direction, distance in movements:
        if direction == 'U':
            new_y = current_y + distance
            for y in range(current_y + 1, new_y + 1):
                if (current_x, y) in houses:
                    visited_houses.add((current_x, y))
            current_y = new_y
        elif direction == 'D':
            new_y = current_y - distance
            for y in range(current_y - 1, new_y - 1, -1):
                if (current_x, y) in houses:
                    visited_houses.add((current_x, y))
            current_y = new_y
        elif direction == 'L':
            new_x = current_x - distance
            for x in range(current_x - 1, new_x - 1, -1):
                if (x, current_y) in houses:
                    visited_houses.add((x, current_y))
            current_x = new_x
        elif direction == 'R':
            new_x = current_x + distance
            for x in range(current_x + 1, new_x + 1):
                if (x, current_y) in houses:
                    visited_houses.add((x, current_y))
            current_x = new_x
    
    # Final position and count of distinct houses visited
    print(current_x, current_y, len(visited_houses))

if __name__ == "__main__":
    main()