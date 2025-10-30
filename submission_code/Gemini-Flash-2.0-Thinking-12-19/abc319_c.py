import collections

def solve():
    grid = []
    for _ in range(3):
        grid.append(list(map(int, input().split())))
    
    lines_positions = []
    # Horizontal lines
    for i in range(3):
        lines_positions.append([(i, 0), (i, 1), (i, 2)])
    # Vertical lines
    for j in range(3):
        lines_positions.append([(0, j), (1, j), (2, j)])
    # Diagonal lines
    lines_positions.append([(0, 0), (1, 1), (2, 2)])
    lines_positions.append([(0, 2), (1, 1), (2, 0)])
    
    line_values = []
    for positions in lines_positions:
        values = [grid[r][c] for r, c in positions]
        line_values.append(values)
        
    case2_lines_indices = []
    for i in range(len(line_values)):
        values = line_values[i]
        counts = collections.Counter(values)
        if len(counts) == 2:
            for count in counts.values():
                if count == 2:
                    case2_lines_indices.append(i)
                    break
                    
    num_permutations = 0
    num_safe_permutations = 0
    
    positions_indices = [(i, j) for i in range(3) for j in range(3)]
    import itertools
    
    all_permutations = list(itertools.permutations(positions_indices))
    num_permutations = len(all_permutations)
    
    for order in all_permutations:
        is_disappointed = False
        for line_index in case2_lines_indices:
            line_pos = lines_positions[line_index]
            line_val = line_values[line_index]
            pos_order_in_line = []
            order_indices_in_line = []
            for pos in line_pos:
                try:
                    index_in_order = order.index(pos)
                    order_indices_in_line.append((index_in_order, pos))
                except ValueError:
                    pass
            order_indices_in_line.sort(key=lambda x: x[0])
            ordered_line_positions = [pos for index, pos in order_indices_in_line]
            ordered_line_values = [grid[r][c] for r, c in ordered_line_positions]
            
            if len(ordered_line_values) >= 3:
                if ordered_line_values[0] == ordered_line_values[1] and ordered_line_values[0] != ordered_line_values[2]:
                    is_disappointed = True
                    break
                    
        if not is_disappointed:
            num_safe_permutations += 1
            
    probability = num_safe_permutations / num_permutations
    print(f"{probability:.20f}")

if __name__ == '__main__':
    solve()