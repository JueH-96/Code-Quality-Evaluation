# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # Read the first line
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
        houses.add((X, Y))
        index += 2
    
    # Read the movement instructions
    movements = []
    for _ in range(M):
        D = data[index]
        C = int(data[index + 1])
        movements.append((D, C))
        index += 2
    
    # Initialize Santa's position
    current_x = S_x
    current_y = S_y
    
    # Set to track visited houses
    visited_houses = set()
    
    # Process each movement
    for D, C in movements:
        if D == 'U':
            for step in range(1, C + 1):
                if (current_x, current_y + step) in houses:
                    visited_houses.add((current_x, current_y + step))
            current_y += C
        elif D == 'D':
            for step in range(1, C + 1):
                if (current_x, current_y - step) in houses:
                    visited_houses.add((current_x, current_y - step))
            current_y -= C
        elif D == 'L':
            for step in range(1, C + 1):
                if (current_x - step, current_y) in houses:
                    visited_houses.add((current_x - step, current_y))
            current_x -= C
        elif D == 'R':
            for step in range(1, C + 1):
                if (current_x + step, current_y) in houses:
                    visited_houses.add((current_x + step, current_y))
            current_x += C
    
    # Final position and number of distinct houses visited
    final_x = current_x
    final_y = current_y
    distinct_houses_count = len(visited_houses)
    
    # Output the result
    print(final_x, final_y, distinct_houses_count)

if __name__ == "__main__":
    main()