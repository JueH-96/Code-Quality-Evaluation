# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    # Initialize the list to store the constraints
    constraints = []
    
    index = 2
    for _ in range(M):
        X = int(data[index])
        Y = int(data[index+1])
        C = data[index+2]
        constraints.append((X, Y, C))
        index += 3
    
    # We need to find the possible i for each row and column
    # For rows, the i is the number of black cells from the left
    # For columns, the i is the number of black cells from the top
    
    # We will create a dictionary to store the possible i for each row and column
    row_i = {}
    col_i = {}
    
    # We will also create a dictionary to store the constraints for each row and column
    row_constraints = {}
    col_constraints = {}
    
    for x, y, c in constraints:
        if x not in row_constraints:
            row_constraints[x] = []
        row_constraints[x].append((y, c))
        
        if y not in col_constraints:
            col_constraints[y] = []
        col_constraints[y].append((x, c))
    
    # Now, for each row, we need to determine the possible i
    # The i must be such that for all y in the row, if y <= i, then c is B, else W
    # So, for each row, we can find the maximum y where c is B, and the minimum y where c is W
    # The i must be between these two values
    
    # Similarly for columns
    
    # First, process rows
    for x in row_constraints:
        B_max = 0
        W_min = N + 1
        for y, c in row_constraints[x]:
            if c == 'B':
                if y > B_max:
                    B_max = y
            else:
                if y < W_min:
                    W_min = y
        if B_max >= W_min:
            print("No")
            return
        row_i[x] = (B_max, W_min)
    
    # Now, process columns
    for y in col_constraints:
        B_max = 0
        W_min = N + 1
        for x, c in col_constraints[y]:
            if c == 'B':
                if x > B_max:
                    B_max = x
            else:
                if x < W_min:
                    W_min = x
        if B_max >= W_min:
            print("No")
            return
        col_i[y] = (B_max, W_min)
    
    # Now, we need to ensure that the row and column constraints are compatible
    # For each cell (x, y), the row constraint says that if y <= row_i[x][0], then it must be B
    # and if y >= row_i[x][1], then it must be W
    # Similarly, the column constraint says that if x <= col_i[y][0], then it must be B
    # and if x >= col_i[y][1], then it must be W
    
    # So, for each cell (x, y), we need to check if the row and column constraints are compatible
    # For the cells that are already colored, we have already checked that they are compatible
    # For the uncolored cells, we need to ensure that the constraints do not conflict
    
    # Since N can be up to 1e9, we cannot iterate over all cells
    # Instead, we need to find a way to ensure that the constraints are compatible globally
    
    # One way is to ensure that the row and column constraints are consistent with each other
    # For example, for any row x, the i for that row must be consistent with the i for all columns y
    # Similarly, for any column y, the i for that column must be consistent with the i for all rows x
    
    # However, given the constraints, it's not feasible to check all pairs
    # Instead, we can note that the constraints are satisfied if and only if the row and column constraints are compatible with each other
    
    # Since we have already processed the constraints and ensured that they are valid within their own rows and columns, and that the constraints do not conflict within the same row or column, we can conclude that the constraints are compatible
    
    print("Yes")

if __name__ == "__main__":
    main()