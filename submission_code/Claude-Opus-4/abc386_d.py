# YOUR CODE HERE
def solve():
    N, M = map(int, input().split())
    
    # min_r[i] = minimum value for r[i], max_r[i] = maximum value for r[i]
    # min_c[j] = minimum value for c[j], max_c[j] = maximum value for c[j]
    min_r = {}
    max_r = {}
    min_c = {}
    max_c = {}
    
    cells = []
    for _ in range(M):
        parts = input().split()
        x, y = int(parts[0]), int(parts[1])
        color = parts[2]
        cells.append((x, y, color))
        
        if color == 'B':
            # Black cell at (x,y) means r[x] >= y and c[y] >= x
            if x not in min_r:
                min_r[x] = y
            else:
                min_r[x] = max(min_r[x], y)
            
            if y not in min_c:
                min_c[y] = x
            else:
                min_c[y] = max(min_c[y], x)
        else:  # color == 'W'
            # White cell at (x,y) means r[x] < y or c[y] < x
            # We'll handle this after collecting all constraints
            pass
    
    # For white cells, we need r[x] < y OR c[y] < x
    # This is harder to handle directly, so let's check consistency
    
    # First, handle white cells
    for x, y, color in cells:
        if color == 'W':
            # We need r[x] < y OR c[y] < x
            # If we already know r[x] >= y from black cells, then we must have c[y] < x
            if x in min_r and min_r[x] >= y:
                if y not in max_c:
                    max_c[y] = x - 1
                else:
                    max_c[y] = min(max_c[y], x - 1)
            # If we already know c[y] >= x from black cells, then we must have r[x] < y
            elif y in min_c and min_c[y] >= x:
                if x not in max_r:
                    max_r[x] = y - 1
                else:
                    max_r[x] = min(max_r[x], y - 1)
            else:
                # We have a choice, but we need to ensure at least one constraint is satisfied
                # We'll check this later
                pass
    
    # Check consistency of constraints
    for row in min_r:
        if row in max_r and min_r[row] > max_r[row]:
            print("No")
            return
    
    for col in min_c:
        if col in max_c and min_c[col] > max_c[col]:
            print("No")
            return
    
    # Now verify all cells are consistent with current constraints
    for x, y, color in cells:
        if color == 'B':
            # Already handled above
            pass
        else:  # color == 'W'
            # Check if at least one of the conditions is satisfied
            r_ok = False
            c_ok = False
            
            # Check if r[x] < y is possible
            if x not in min_r or min_r[x] < y:
                r_ok = True
            
            # Check if c[y] < x is possible
            if y not in min_c or min_c[y] < x:
                c_ok = True
            
            if not r_ok and not c_ok:
                print("No")
                return
    
    print("Yes")

solve()