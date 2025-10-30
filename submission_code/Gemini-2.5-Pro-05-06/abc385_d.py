import collections
import bisect
import sys

def main():
    # Use functions for faster I/O with large inputs
    input = sys.stdin.readline

    N, M, S_x, S_y = map(int, input().split())

    houses_x_map = collections.defaultdict(list)
    houses_y_map = collections.defaultdict(list)
    
    if N > 0: # Optimization: only process houses if there are any
        for _ in range(N):
            X, Y = map(int, input().split())
            houses_x_map[X].append(Y)
            houses_y_map[Y].append(X)

        # Sort coordinate lists for binary search
        for x_coord in houses_x_map:
            houses_x_map[x_coord].sort()
        
        for y_coord in houses_y_map:
            houses_y_map[y_coord].sort()

    curr_x, curr_y = S_x, S_y
    
    visited_houses_count = 0
    # Set to keep track of unique visited houses
    _visited_houses_set = set() 

    for _ in range(M):
        D_str, C_str = input().split()
        C = int(C_str)
        
        prev_x, prev_y = curr_x, curr_y

        if D_str == 'U':
            curr_y += C
            # Check if any houses are on the vertical line X = curr_x
            if curr_x in houses_x_map:
                y_coords_on_line = houses_x_map[curr_x]
                
                # Segment Y-coords: [prev_y, curr_y] (since U means prev_y < curr_y)
                # bisect_left finds insertion point for prev_y (first Y >= prev_y)
                idx_start = bisect.bisect_left(y_coords_on_line, prev_y)
                # bisect_right finds insertion point for curr_y (first Y > curr_y)
                idx_end = bisect.bisect_right(y_coords_on_line, curr_y)
                
                for i in range(idx_start, idx_end):
                    house_y_coord = y_coords_on_line[i]
                    # If house not already visited, add to set and increment count
                    if (curr_x, house_y_coord) not in _visited_houses_set:
                        _visited_houses_set.add((curr_x, house_y_coord))
                        visited_houses_count +=1
                    
        elif D_str == 'D':
            curr_y -= C
            if curr_x in houses_x_map:
                y_coords_on_line = houses_x_map[curr_x]
                
                # Segment Y-coords: [curr_y, prev_y] (since D means curr_y < prev_y)
                idx_start = bisect.bisect_left(y_coords_on_line, curr_y)
                idx_end = bisect.bisect_right(y_coords_on_line, prev_y)
                
                for i in range(idx_start, idx_end):
                    house_y_coord = y_coords_on_line[i]
                    if (curr_x, house_y_coord) not in _visited_houses_set:
                        _visited_houses_set.add((curr_x, house_y_coord))
                        visited_houses_count +=1

        elif D_str == 'L':
            curr_x -= C
            # Check if any houses are on the horizontal line Y = curr_y
            if curr_y in houses_y_map:
                x_coords_on_line = houses_y_map[curr_y]
                
                # Segment X-coords: [curr_x, prev_x] (since L means curr_x < prev_x)
                idx_start = bisect.bisect_left(x_coords_on_line, curr_x)
                idx_end = bisect.bisect_right(x_coords_on_line, prev_x)
                
                for i in range(idx_start, idx_end):
                    house_x_coord = x_coords_on_line[i]
                    if (house_x_coord, curr_y) not in _visited_houses_set:
                        _visited_houses_set.add((house_x_coord, curr_y))
                        visited_houses_count +=1
                    
        elif D_str == 'R':
            curr_x += C
            if curr_y in houses_y_map:
                x_coords_on_line = houses_y_map[curr_y]

                # Segment X-coords: [prev_x, curr_x] (since R means prev_x < curr_x)
                idx_start = bisect.bisect_left(x_coords_on_line, prev_x)
                idx_end = bisect.bisect_right(x_coords_on_line, curr_x)
                
                for i in range(idx_start, idx_end):
                    house_x_coord = x_coords_on_line[i]
                    if (house_x_coord, curr_y) not in _visited_houses_set:
                        _visited_houses_set.add((house_x_coord, curr_y))
                        visited_houses_count +=1
                        
    sys.stdout.write(f"{curr_x} {curr_y} {visited_houses_count}
")

if __name__ == '__main__':
    main()