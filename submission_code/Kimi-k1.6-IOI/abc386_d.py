def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    M = int(input[idx])
    idx += 1
    
    max_b_row = dict()  # key: row x, value: max y in B cells of x
    max_b_col = dict()  # key: column y, value: max x in B cells of y
    w_cells = []
    
    for _ in range(M):
        X = int(input[idx])
        idx += 1
        Y = int(input[idx])
        idx += 1
        C = input[idx]
        idx += 1
        
        if C == 'B':
            # Update row X's max Y
            if X in max_b_row:
                if Y > max_b_row[X]:
                    max_b_row[X] = Y
            else:
                max_b_row[X] = Y
            # Update column Y's max X
            if Y in max_b_col:
                if X > max_b_col[Y]:
                    max_b_col[Y] = X
            else:
                max_b_col[Y] = X
        else:
            w_cells.append((X, Y))
    
    possible = True
    for x, y in w_cells:
        lx = max_b_row.get(x, 0)
        my = max_b_col.get(y, 0)
        if lx > 0 and my > 0:
            if not (lx < y or my < x):
                possible = False
                break
    
    print("Yes" if possible else "No")

if __name__ == "__main__":
    main()