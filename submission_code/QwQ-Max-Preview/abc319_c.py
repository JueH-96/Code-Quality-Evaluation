import itertools

def main():
    grid = []
    for _ in range(3):
        row = list(map(int, input().split()))
        grid.append(row)
    
    # Precompute all lines (rows, columns, diagonals)
    lines = [
        # Rows
        [(0, 0), (0, 1), (0, 2)],
        [(1, 0), (1, 1), (1, 2)],
        [(2, 0), (2, 1), (2, 2)],
        # Columns
        [(0, 0), (1, 0), (2, 0)],
        [(0, 1), (1, 1), (2, 1)],
        [(0, 2), (1, 2), (2, 2)],
        # Diagonals
        [(0, 0), (1, 1), (2, 2)],
        [(0, 2), (1, 1), (2, 0)],
    ]
    
    # Identify risky lines (exactly two same and one different)
    risky_lines = []
    for line in lines:
        values = [grid[i][j] for (i, j) in line]
        if len(set(values)) != 2:
            continue
        # Find the third cell (the one with different value)
        # Count occurrences
        count = {}
        for v in values:
            count[v] = count.get(v, 0) + 1
        # Find the value that appears twice and the one that appears once
        two_val = [k for k, v in count.items() if v == 2][0]
        one_val = [k for k, v in count.items() if v == 1][0]
        # Find the index of the third cell (the one with one_val)
        third_cell = line[values.index(one_val)]
        # The other two are the same
        risky_lines.append({
            'cells': line,
            'third_cell': third_cell,
            'other_cells': [cell for cell in line if grid[cell[0]][cell[1]] == two_val]
        })
    
    n_risky = len(risky_lines)
    total = 0
    # Generate all subsets of risky_lines using inclusion-exclusion
    for mask in range(1, 1 << n_risky):
        # Current subset is the bits set in mask
        current_lines = []
        for i in range(n_risky):
            if (mask >> i) & 1:
                current_lines.append(risky_lines[i])
        # Build the constraints: for each line in current_lines, third_cell must be after other_cells
        # Collect all constraints (a, b) meaning a must come before b
        constraints = []
        # Also collect all involved cells
        involved_cells = set()
        for line in current_lines:
            third = line['third_cell']
            others = line['other_cells']
            for other in others:
                constraints.append((other, third))
            involved_cells.update(others)
            involved_cells.add(third)
        involved_cells = list(involved_cells)
        m = len(involved_cells)
        # Check if there's a cycle in constraints
        # Build a DAG
        adj = {}
        in_degree = {}
        for a, b in constraints:
            if a not in adj:
                adj[a] = []
            if b not in in_degree:
                in_degree[b] = 0
            adj[a].append(b)
            in_degree[b] = in_degree.get(b, 0) + 1
        # Check for cycles
        # Perform topological sort to check for cycles
        # Use Kahn's algorithm
        q = []
        for cell in involved_cells:
            if in_degree.get(cell, 0) == 0:
                q.append(cell)
        processed = 0
        while q:
            u = q.pop(0)
            processed += 1
            if u not in adj:
                continue
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)
        if processed != len(involved_cells):
            # There's a cycle, this subset is impossible
            continue
        # Now compute the number of topological orderings of involved_cells
        # and multiply by the factorial of the remaining cells
        # This part is complex, but for small m (<= 8), we can generate all permutations of involved_cells and count valid ones
        # However, this is not feasible for m > 10, but given the problem constraints, it's manageable
        # For the problem's constraints, m can be up to 8 (if all lines are risky), but this is unlikely
        # Generate all permutations of involved_cells and count those that satisfy all constraints
        valid_orders = 0
        for perm in itertools.permutations(involved_cells):
            valid = True
            for a, b in constraints:
                if perm.index(a) > perm.index(b):
                    valid = False
                    break
            if valid:
                valid_orders += 1
        # The number of valid permutations is valid_orders * factorial(9 - m) * (factorial(m) / factorial(m)) ??
        # The total permutations is 9! = 362880
        # The valid_orders is the number of valid permutations of the involved_cells, and the rest can be arranged freely
        # So the total for this subset is valid_orders * factorial(9 - m)
        # But the involved_cells are part of the 9 cells, so the total number is valid_orders * factorial(9 - m) * (number of ways to interleave)
        # Wait, no. The valid_orders is the number of permutations of the involved_cells that meet the constraints. The remaining cells can be placed anywhere else in the permutation.
        # The correct way is to compute the number of ways to choose the positions for the involved_cells and the remaining cells, and multiply by valid_orders and factorial(remaining)
        # The number of ways is C(9, m) * valid_orders * factorial(9 - m)
        # C(9, m) is the number of ways to choose positions for the involved_cells
        # valid_orders is the number of valid permutations of the involved_cells
        # factorial(9 - m) is the number of permutations of the remaining cells
        # But this is equivalent to 9! * valid_orders / (factorial(m) * factorial(9 - m)) ) * factorial(9 - m) ) = 9! * valid_orders / factorial(m)
        # Because C(9, m) * valid_orders * factorial(9 - m) = (9! / (m! (9-m)! )) ) * valid_orders * (9-m)! ) = 9! * valid_orders / m!
        # So the number of permutations for this subset is (9! / m! ) * valid_orders
        # But valid_orders is the number of valid permutations of the involved_cells, which is m! multiplied by the probability of valid permutations.
        # So the total is 9! * (valid_orders / m! )
        count = 362880 * valid_orders // factorial(m)
        # Apply inclusion-exclusion
        if bin(mask).count('1') % 2 == 1:
            total += count
        else:
            total -= count
    
    good = 362880 - total
    prob = good / 362880
    print("{0:.20f}".format(prob))

def factorial(n):
    if n == 0:
        return 1
    res = 1
    for i in range(2, n+1):
        res *= i
    return res

if __name__ == "__main__":
    main()