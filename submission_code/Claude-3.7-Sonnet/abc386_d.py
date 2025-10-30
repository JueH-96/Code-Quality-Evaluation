def can_satisfy_conditions(N, colored_cells):
    # Extract the rows and columns that are explicitly involved
    rows = set([x for x, _, _ in colored_cells])
    cols = set([y for _, y, _ in colored_cells])
    
    # Initialize constraints
    R_min = {i: 0 for i in rows}
    R_max = {i: N for i in rows}
    C_min = {j: 0 for j in cols}
    C_max = {j: N for j in cols}
    
    # Process black cells
    for x, y, color in colored_cells:
        if color == 'B':
            R_min[x] = max(R_min[x], y)
            C_min[y] = max(C_min[y], x)
    
    # Process white cells
    for x, y, color in colored_cells:
        if color == 'W':
            R_max[x] = min(R_max[x], y - 1)
            C_max[y] = min(C_max[y], x - 1)
    
    # Check if constraints are feasible
    for i in rows:
        if R_min[i] > R_max[i]:
            return False
    for j in cols:
        if C_min[j] > C_max[j]:
            return False
    
    return True

if __name__ == "__main__":
    N, M = map(int, input().split())
    colored_cells = []
    for _ in range(M):
        x, y, c = input().split()
        colored_cells.append((int(x), int(y), c))
    
    if can_satisfy_conditions(N, colored_cells):
        print("Yes")
    else:
        print("No")