def read_sheet(h, w):
    grid = []
    for _ in range(h):
        grid.append(input().strip())
    return grid

def get_blacks(sheet, h, w):
    blacks = []
    for i in range(h):
        for j in range(w):
            if sheet[i][j] == '#':
                blacks.append((i, j))
    return blacks

def main():
    # Read A
    H_A, W_A = map(int, input().split())
    A = read_sheet(H_A, W_A)
    A_blacks = get_blacks(A, H_A, W_A)
    
    # Read B
    H_B, W_B = map(int, input().split())
    B = read_sheet(H_B, W_B)
    B_blacks = get_blacks(B, H_B, W_B)
    
    # Read X
    H_X, W_X = map(int, input().split())
    X = read_sheet(H_X, W_X)
    
    # Find min and max rows and columns for blacks in A and B
    min_row_A = min(i for i, _ in A_blacks)
    max_row_A = max(i for i, _ in A_blacks)
    min_col_A = min(j for _, j in A_blacks)
    max_col_A = max(j for _, j in A_blacks)
    
    min_row_B = min(i for i, _ in B_blacks)
    max_row_B = max(i for i, _ in B_blacks)
    min_col_B = min(j for _, j in B_blacks)
    max_col_B = max(j for _, j in B_blacks)
    
    # Determine ranges for placing A and B within X
    r_a_min = max(0, -min_row_A)
    r_a_max = min(H_X - max_row_A - 1, H_X - H_A + 1)
    c_a_min = max(0, -min_col_A)
    c_a_max = min(W_X - max_col_A - 1, W_X - W_A + 1)
    
    r_b_min = max(0, -min_row_B)
    r_b_max = min(H_X - max_row_B - 1, H_X - H_B + 1)
    c_b_min = max(0, -min_col_B)
    c_b_max = min(W_X - max_col_B - 1, W_X - W_B + 1)
    
    # Iterate over all possible placements of A and B within X
    for r_a in range(r_a_min, r_a_max + 1):
        for c_a in range(c_a_min, c_a_max + 1):
            for r_b in range(r_b_min, r_b_max + 1):
                for c_b in range(c_b_min, c_b_max + 1):
                    # Create the union grid
                    union = [['.' for _ in range(W_X)] for _ in range(H_X)]
                    # Place A
                    for i in range(H_A):
                        for j in range(W_A):
                            if A[i][j] == '#':
                                rr = r_a + i
                                cc = c_a + j
                                union[rr][cc] = '#'
                    # Place B
                    for i in range(H_B):
                        for j in range(W_B):
                            if B[i][j] == '#':
                                rr = r_b + i
                                cc = c_b + j
                                union[rr][cc] = '#'
                    # Check if union matches X
                    matches = True
                    for i in range(H_X):
                        for j in range(W_X):
                            if union[i][j] != X[i][j]:
                                matches = False
                                break
                        if not matches:
                            break
                    if matches:
                        print("Yes")
                        return
    print("No")

if __name__ == "__main__":
    main()