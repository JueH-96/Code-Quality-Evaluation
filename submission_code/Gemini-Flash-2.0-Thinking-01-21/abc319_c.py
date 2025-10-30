import collections

def solve():
    c = []
    for _ in range(3):
        c.append(list(map(int, input().split())))
    
    lines_cells = []
    lines_values = []
    
    # Horizontal lines
    for i in range(3):
        cells = []
        values = []
        for j in range(3):
            cells.append((i, j))
            values.append(c[i][j])
        lines_cells.append(cells)
        lines_values.append(values)
        
    # Vertical lines
    for j in range(3):
        cells = []
        values = []
        for i in range(3):
            cells.append((i, j))
            values.append(c[i][j])
        lines_cells.append(cells)
        lines_values.append(values)
        
    # Diagonals
    diag1_cells = []
    diag1_values = []
    for i in range(3):
        diag1_cells.append((i, i))
        diag1_values.append(c[i][i])
    lines_cells.append(diag1_cells)
    lines_values.append(diag1_values)
    
    diag2_cells = []
    diag2_values = []
    for i in range(3):
        diag2_cells.append((i, 2 - i))
        diag2_values.append(c[i][2 - i])
    lines_cells.append(diag2_cells)
    lines_values.append(diag2_values)
    
    bad_orders_for_line = []
    for values in lines_values:
        value_counts = collections.Counter(values)
        type2_line = False
        for val in value_counts:
            if value_counts[val] == 2:
                type2_line = True
                break
        if type2_line:
            val1 = -1
            val2 = -1
            unique_vals = sorted(list(set(values)))
            val1 = unique_vals[0]
            val2 = unique_vals[1]
            if value_counts[val1] == 2:
                same_val = val1
                diff_val = val2
            else:
                same_val = val2
                diff_val = val1
            
            line_bad_orders = []
            indices = list(range(3))
            import itertools
            for order_indices in itertools.permutations(indices):
                order_values = [values[i] for i in order_indices]
                if order_values[0] == order_values[1] and order_values[0] != order_values[2]:
                    line_bad_orders.append(tuple(order_indices))
            bad_orders_for_line.append(line_bad_orders)
        else:
            bad_orders_for_line.append([])
            
    total_permutations = 0
    good_permutations = 0
    
    all_cell_indices = []
    for i in range(3):
        for j in range(3):
            all_cell_indices.append((i, j))
            
    import itertools
    
    for order_of_indices in itertools.permutations(all_cell_indices):
        total_permutations += 1
        is_disappointed = False
        for line_index in range(8):
            line_cells = lines_cells[line_index]
            line_values = lines_values[line_index]
            bad_orders = bad_orders_for_line[line_index]
            if not bad_orders:
                continue
                
            indices_in_permutation = []
            for cell in line_cells:
                try:
                    indices_in_permutation.append(order_of_indices.index(cell))
                except ValueError:
                    pass 
            
            sorted_indices_in_permutation = sorted(indices_in_permutation)
            seen_order_indices_in_line = []
            for index in sorted_indices_in_permutation:
                seen_order_indices_in_line.append(line_cells.index(order_of_indices[index]))
                
            if tuple(seen_order_indices_in_line) in bad_orders:
                is_disappointed = True
                break
                
        if not is_disappointed:
            good_permutations += 1
            
    probability = good_permutations / total_permutations
    print(f"{probability:.20f}")

if __name__ == '__main__':
    solve()