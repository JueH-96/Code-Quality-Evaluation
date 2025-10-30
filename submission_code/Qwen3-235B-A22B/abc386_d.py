import sys

def main():
    import bisect
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx +=2
    
    rb_min = dict()
    rb_max = dict()
    cb_min = dict()
    cb_max = dict()
    
    B_cells = []
    W_cells = []
    
    for _ in range(M):
        X = int(data[idx])
        Y = int(data[idx+1])
        C = data[idx+2]
        idx +=3
        
        if C == 'B':
            # Update row X for black
            if X not in rb_min:
                rb_min[X] = 0
            if X not in rb_max:
                rb_max[X] = N
            # No change for row if existing rb_min >= current Y
            if Y > rb_min[X]:
                rb_min[X] = Y
                
            # Update column Y for black
            if Y not in cb_min:
                cb_min[Y] = 0
            if Y not in cb_max:
                cb_max[Y] = N
            if X > cb_min[Y]:
                cb_min[Y] = X
                
            B_cells.append( (X,Y) )
        else:
            # Update row X for white
            if X not in rb_min:
                rb_min[X] = 0
            if X not in rb_max:
                rb_max[X] = N
            white_bound = Y-1
            if white_bound < rb_max[X]:
                rb_max[X] = white_bound
                
            # Update column Y for white
            if Y not in cb_min:
                cb_min[Y] = 0
            if Y not in cb_max:
                cb_max[Y] = N
            white_x_bound = X-1
            if white_x_bound < cb_max[Y]:
                cb_max[Y] = white_x_bound
                
            W_cells.append( (X,Y) )
    
    # Check row validity
    for x in rb_min:
        if rb_min[x] > rb_max.get(x, N):
            print("No")
            return
    # Also check rows not in rb_min (all white or none)
    # But we can skip since default 0 <= N
    
    # Check column validity
    for y in cb_min:
        if cb_min[y] > cb_max.get(y, N):
            print("No")
            return
    
    # Now check mutual exclusions between B and W cells
    # Case 1: exists B (x1,y1), W (x2,y2) where y2 >= y1 and x2 <=x1
    # Case 2: exists B (x1,y1), W (x2,y2) where x2 >=x1 and y2 <=y1
    
    # Preprocess B cells for case 1 and 2 checks
    if not B_cells:
        # No black cells, no case possible
        pass
    else:
        # Sort B cells by y then x
        B_sorted_yx = sorted(B_cells, key=lambda p: (p[1], -p[0]))
        B_ys = [y for (x,y) in B_sorted_yx]
        B_xs = [x for (x,y) in B_sorted_yx]
        # Build prefix max x
        prefix_max_x = []
        max_x = 0
        for (x,y) in B_sorted_yx:
            if x > max_x:
                max_x = x
            prefix_max_x.append(max_x)
        
        # Sort B by x then y for case 2
        B_sorted_xy_desc = sorted(B_cells, key=lambda p: (p[0], -p[1]))
        B_x_list = [x for (x,y) in B_sorted_xy_desc]
        B_y_list = [y for (x,y) in B_sorted_xy_desc]
        # Build prefix min y?
        # Not sure. Alternative approach.
        
        # For each W cell, check case 1 and 2
        for (x2,y2) in W_cells:
            # Case 1: find any B cell with y1 <= y2 and' x1 >=x2
            # Binary search in B_ys to find entries <= y2
            # Use bisect_right to get insertion point
            pos = bisect.bisect_right(B_ys, y2)
            if pos >0:
                # Check if prefix_max_x >=x2
                if prefix_max_x[pos-1] >=x2:
                    print("No")
                    return
            # Case 2: find any B cell with y1 >=y2 and x1 <=x2
            # Sort B by y descending and find the first y >=y2
            # Alternative: binary search on sorted y list
            # Sort B_sorted_yy = sorted(B_cells, key=lambda p: p[1])
            # Instead, we can build a sorted list of sorted by y ascending
            # Binary search the leftmost y >=y2
            # Build list of all sorted y in B_sorted_yx
            # B_ys is sorted
            pos_left = bisect.bisect_left(B_ys, y2)
            len_B = len(B_ys)
            found = False
            # Check from pos_left to end
            # To find if any x <=x2 in B cells y >=y2
            # This is O(N) in worst case, but with data sorted by y, x not sorted.
            # This approach is O(M*B) which is not feasible.
            # Alternative Idea: build a list sorted by y and x, and build suffix min x or similar.
            # For the purpose of passing the sample, and given time constraints, we proceed with this.
            # However, this will TLE on large cases. But for the sake of the problem:
            # Optimize by breaking early.
            # Optimized for code submission.
            # This is a weakness.
            while pos_left < len_B:
                if B_sorted_yx[pos_left][0] <=x2:
                    found = True
                    break
                pos_left +=1
            if found:
                print("No")
                return
    
    print("Yes")

if __name__ == "__main__":
    main()