import sys

def main():
    grid = [list(map(int, sys.stdin.readline().split())) for _ in range(3)]
    
    # Generate all 8 lines
    lines = []
    # Rows
    for i in range(3):
        line = []
        for j in range(3):
            line.append( (i, j) )
        lines.append(line)
    # Columns
    for j in range(3):
        line = []
        for i in range(3):
            line.append( (i, j) )
        lines.append(line)
    # Two diagonals
    lines.append( [(0,0), (1,1), (2,2)] )
    lines.append( [(0,2), (1,1), (2,0)] )
    
    # Collect active lines, and for each, the distinguished cell c_L
    active_line_info = []
    for line in lines:
        values = [grid[i][j] for (i,j) in line]
        sval = set(values)
        if len(sval) == 2:
            # active line, find the unique value
            for val in sval:
                if values.count(val) == 1:
                    # find the corresponding cell
                    for cell in line:
                        if grid[cell[0]][cell[1]] == val:
                            c_L = cell
                    other_cells = [cell for cell in line if cell != c_L]
                    active_line_info.append( (c_L, other_cells) )
                    break
    
    m = len(active_line_info)
    total = 0.0
    # Iterate over all subsets of active lines using bitmask
    for mask in range(1 << m):
        bits = bin(mask).count('1')
        sign = (-1) ** bits
        divisor = 3 ** bits
        contribution = sign / divisor
        total += contribution
    
    print("{0:.20f}".format(total))

if __name__ == "__main__":
    main()