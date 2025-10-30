import sys
import math

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    S_x = int(data[idx+2])
    S_y = int(data[idx+3])
    idx += 4
    houses = []
    for _ in range(N):
        X = int(data[idx])
        Y = int(data[idx+1])
        houses.append((X, Y))
        idx += 2
    moves = []
    for _ in range(M):
        D = data[idx]
        C = int(data[idx+1])
        moves.append((D, C))
        idx += 2
    
    current_x, current_y = S_x, S_y
    visited = set()
    
    for D, C in moves:
        if D == 'U':
            new_x, new_y = current_x, current_y + C
        elif D == 'D':
            new_x, new_y = current_x, current_y - C
        elif D == 'L':
            new_x, new_y = current_x - C, current_y
        elif D == 'R':
            new_x, new_y = current_x + C, current_y
        
        # Calculate the line segment from (current_x, current_y) to (new_x, new_y)
        # and check if any house lies on this segment
        # The line equation is (y - y1) = m(x - x1), where m is the slope
        # For vertical lines, x is constant
        # For horizontal lines, y is constant
        # For other lines, we need to check if the house lies on the line and within the segment
        
        if current_x == new_x:
            # Vertical line
            x = current_x
            y_min = min(current_y, new_y)
            y_max = max(current_y, new_y)
            for house in houses:
                hx, hy = house
                if hx == x and y_min <= hy <= y_max:
                    visited.add((hx, hy))
        elif current_y == new_y:
            # Horizontal line
            y = current_y
            x_min = min(current_x, new_x)
            x_max = max(current_x, new_x)
            for house in houses:
                hx, hy = house
                if hy == y and x_min <= hx <= x_max:
                    visited.add((hx, hy))
        else:
            # General line
            # Calculate the slope m and intercept b
            m = (new_y - current_y) / (new_x - current_x)
            b = current_y - m * current_x
            x_min = min(current_x, new_x)
            x_max = max(current_x, new_x)
            y_min = min(current_y, new_y)
            y_max = max(current_y, new_y)
            for house in houses:
                hx, hy = house
                if x_min <= hx <= x_max and y_min <= hy <= y_max:
                    # Check if the house lies on the line
                    if abs(hy - (m * hx + b)) < 1e-9:
                        visited.add((hx, hy))
        
        current_x, current_y = new_x, new_y
    
    # Now, count the number of distinct houses in visited
    count = len(visited)
    print(current_x, current_y, count)

if __name__ == "__main__":
    main()