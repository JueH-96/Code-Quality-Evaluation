from itertools import product

def solve():
    n, m = map(int, input().split())
    
    if m == 0:
        return "Yes"
    
    constraints = []
    rows = set()
    cols = set()
    
    for _ in range(m):
        x, y, c = input().split()
        x, y = int(x), int(y)
        constraints.append((x, y, c))
        rows.add(x)
        cols.add(y)
    
    # Initialize bounds
    row_lb = {r: 0 for r in rows}
    row_ub = {r: n for r in rows}
    col_lb = {c: 0 for c in cols}
    col_ub = {c: n for c in cols}
    
    # Process black cells - set lower bounds
    for x, y, c in constraints:
        if c == 'B':
            row_lb[x] = max(row_lb[x], y)
            col_lb[y] = max(col_lb[y], x)
    
    # Process white cells with constraint propagation
    changed = True
    while changed:
        changed = False
        for x, y, c in constraints:
            if c == 'W':
                # If row_lb[x] >= y, then we must have col_black[y] < x
                if row_lb[x] >= y:
                    new_ub = min(col_ub[y], x - 1)
                    if new_ub < col_lb[y]:
                        return "No"
                    if new_ub < col_ub[y]:
                        col_ub[y] = new_ub
                        changed = True
                
                # If col_lb[y] >= x, then we must have row_black[x] < y
                if col_lb[y] >= x:
                    new_ub = min(row_ub[x], y - 1)
                    if new_ub < row_lb[x]:
                        return "No"
                    if new_ub < row_ub[x]:
                        row_ub[x] = new_ub
                        changed = True
    
    # Check if bounds are consistent
    for r in rows:
        if row_lb[r] > row_ub[r]:
            return "No"
    for c in cols:
        if col_lb[c] > col_ub[c]:
            return "No"
    
    # Enumerate assignments within bounds
    row_list = list(rows)
    col_list = list(cols)
    
    row_ranges = [list(range(row_lb[r], row_ub[r] + 1)) for r in row_list]
    col_ranges = [list(range(col_lb[c], col_ub[c] + 1)) for c in col_list]
    
    for row_assignment in product(*row_ranges):
        for col_assignment in product(*col_ranges):
            row_black = {row_list[i]: row_assignment[i] for i in range(len(row_list))}
            col_black = {col_list[i]: col_assignment[i] for i in range(len(col_list))}
            
            # Check all constraints
            valid = True
            for x, y, c in constraints:
                is_black = (y <= row_black[x]) and (x <= col_black[y])
                if (c == 'B' and not is_black) or (c == 'W' and is_black):
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check global consistency
            globally_consistent = True
            for r in rows:
                for c in cols:
                    row_says_black = (c <= row_black[r])
                    col_says_black = (r <= col_black[c])
                    if row_says_black != col_says_black:
                        globally_consistent = False
                        break
                if not globally_consistent:
                    break
            
            if globally_consistent:
                return "Yes"
    
    return "No"

print(solve())