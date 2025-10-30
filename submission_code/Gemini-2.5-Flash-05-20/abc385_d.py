import sys
from collections import defaultdict
from bisect import bisect_left, bisect_right

def solve():
    # Read N, M, S_x, S_y from the first line
    N, M, S_x, S_y = map(int, sys.stdin.readline().split())

    # Dictionaries to store houses, indexed by X or Y coordinate.
    # Values will be lists of the other coordinate.
    houses_by_x = defaultdict(list)
    houses_by_y = defaultdict(list)

    # Read N house coordinates and populate the dictionaries
    for _ in range(N):
        X_i, Y_i = map(int, sys.stdin.readline().split())
        houses_by_x[X_i].append(Y_i)
        houses_by_y[Y_i].append(X_i)

    # Sort the lists of coordinates within each dictionary value.
    # This is crucial for efficient range queries using binary search (bisect_left/right).
    for x_coord in houses_by_x:
        houses_by_x[x_coord].sort()
    for y_coord in houses_by_y:
        houses_by_y[y_coord].sort()

    # Initialize Santa's current position
    current_x, current_y = S_x, S_y
    
    # Use a set to store distinct houses visited. Tuples (x, y) are hashable.
    visited_houses = set()

    # Process M movement instructions
    for _ in range(M):
        direction, C_str = sys.stdin.readline().split()
        C = int(C_str)

        # Store Santa's position before the current move
        prev_x, prev_y = current_x, current_y
        
        # Determine the constant coordinate and the range for the varying coordinate
        if direction == 'U':
            current_y += C
            # Vertical movement: x-coordinate is constant
            line_coord_fixed = prev_x
            
            # Check if there are any houses on this specific x-line
            if line_coord_fixed in houses_by_x:
                # Get the sorted list of y-coordinates for houses on this x-line
                coords_on_line = houses_by_x[line_coord_fixed]
                
                # Determine the range of y-coordinates covered by the movement
                coord_min_range = min(prev_y, current_y)
                coord_max_range = max(prev_y, current_y)
                
                # Find the indices of houses whose y-coordinates are within the range
                idx_start = bisect_left(coords_on_line, coord_min_range)
                idx_end = bisect_right(coords_on_line, coord_max_range)
                
                # Add all found houses in this range to the set of visited houses
                for i in range(idx_start, idx_end):
                    visited_houses.add((line_coord_fixed, coords_on_line[i]))

        elif direction == 'D':
            current_y -= C
            line_coord_fixed = prev_x
            if line_coord_fixed in houses_by_x:
                coords_on_line = houses_by_x[line_coord_fixed]
                coord_min_range = min(prev_y, current_y)
                coord_max_range = max(prev_y, current_y)
                idx_start = bisect_left(coords_on_line, coord_min_range)
                idx_end = bisect_right(coords_on_line, coord_max_range)
                for i in range(idx_start, idx_end):
                    visited_houses.add((line_coord_fixed, coords_on_line[i]))

        elif direction == 'L':
            current_x -= C
            # Horizontal movement: y-coordinate is constant
            line_coord_fixed = prev_y
            
            # Check if there are any houses on this specific y-line
            if line_coord_fixed in houses_by_y:
                # Get the sorted list of x-coordinates for houses on this y-line
                coords_on_line = houses_by_y[line_coord_fixed]
                
                # Determine the range of x-coordinates covered by the movement
                coord_min_range = min(prev_x, current_x)
                coord_max_range = max(prev_x, current_x)
                
                # Find the indices of houses whose x-coordinates are within the range
                idx_start = bisect_left(coords_on_line, coord_min_range)
                idx_end = bisect_right(coords_on_line, coord_max_range)
                
                # Add all found houses in this range to the set of visited houses
                for i in range(idx_start, idx_end):
                    visited_houses.add((coords_on_line[i], line_coord_fixed))

        elif direction == 'R':
            current_x += C
            line_coord_fixed = prev_y
            if line_coord_fixed in houses_by_y:
                coords_on_line = houses_by_y[line_coord_fixed]
                coord_min_range = min(prev_x, current_x)
                coord_max_range = max(prev_x, current_x)
                idx_start = bisect_left(coords_on_line, coord_min_range)
                idx_end = bisect_right(coords_on_line, coord_max_range)
                for i in range(idx_start, idx_end):
                    visited_houses.add((coords_on_line[i], line_coord_fixed))
        
    # Print the final position and the count of distinct visited houses
    print(current_x, current_y, len(visited_houses))

# Call the solve function to execute the program
solve()