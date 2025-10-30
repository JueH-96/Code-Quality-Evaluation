# YOUR CODE HERE
n, m = map(int, input().split())

cells = []
for _ in range(m):
    parts = input().split()
    x, y, c = int(parts[0]), int(parts[1]), parts[2]
    cells.append((x, y, c))

if m == 0:
    print("Yes")
else:
    # For each row r, track constraints: lower_bound <= b(r) <= upper_bound
    constraints = {}
    
    for x, y, c in cells:
        if x not in constraints:
            constraints[x] = (0, n)
        
        if c == 'B':
            # Black cell: b(x) >= y
            constraints[x] = (max(constraints[x][0], y), constraints[x][1])
        else:
            # White cell: b(x) <= y-1
            constraints[x] = (constraints[x][0], min(constraints[x][1], y - 1))
    
    # Check if constraints are consistent
    valid = True
    for r, (lb, ub) in constraints.items():
        if lb > ub:
            valid = False
            break
    
    if valid:
        # Check if we can find a non-increasing boundary function
        sorted_rows = sorted(constraints.keys())
        boundary = {}
        
        for i, r in enumerate(sorted_rows):
            lb, ub = constraints[r]
            
            if i == 0:
                boundary[r] = ub
            else:
                prev_r = sorted_rows[i - 1]
                # Ensure non-increasing: b(r) <= b(prev_r)
                boundary[r] = min(ub, boundary[prev_r])
            
            if boundary[r] < lb:
                valid = False
                break
    
    print("Yes" if valid else "No")