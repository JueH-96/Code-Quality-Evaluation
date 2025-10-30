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
    
    # We will represent the constraints as:
    # For each row, the i must be consistent with all the cells in that row
    # Similarly for columns
    
    # We will create a dictionary to store the possible i for each row and column
    # Initially, i can be any value from 0 to N
    
    # But since the constraints are given, we need to narrow down the possible i
    
    # For each row, we will find the minimum and maximum possible i
    # Similarly for each column
    
    # Initialize the row and column constraints
    row_min = {}
    row_max = {}
    col_min = {}
    col_max = {}
    
    for x, y, c in constraints:
        if c == 'B':
            # For row x, the i must be at least y
            if x not in row_min:
                row_min[x] = y
            else:
                row_min[x] = max(row_min[x], y)
            # For column y, the i must be at least x
            if y not in col_min:
                col_min[y] = x
            else:
                col_min[y] = max(col_min[y], x)
        else:
            # For row x, the i must be less than y
            if x not in row_max:
                row_max[x] = y - 1
            else:
                row_max[x] = min(row_max[x], y - 1)
            # For column y, the i must be less than x
            if y not in col_max:
                col_max[y] = x - 1
            else:
                col_max[y] = min(col_max[y], x - 1)
    
    # Now, for each row, we need to find the intersection of the min and max
    # Similarly for each column
    
    # We will create a dictionary to store the final i for each row and column
    row_i = {}
    col_i = {}
    
    # Process rows
    for x in row_min:
        if x in row_max:
            if row_min[x] > row_max[x]:
                print("No")
                return
            row_i[x] = (row_min[x], row_max[x])
        else:
            row_i[x] = (row_min[x], N)
    
    for x in row_max:
        if x not in row_min:
            row_i[x] = (0, row_max[x])
    
    # Process columns
    for y in col_min:
        if y in col_max:
            if col_min[y] > col_max[y]:
                print("No")
                return
            col_i[y] = (col_min[y], col_max[y])
        else:
            col_i[y] = (col_min[y], N)
    
    for y in col_max:
        if y not in col_min:
            col_i[y] = (0, col_max[y])
    
    # Now, we need to ensure that the row and column constraints are consistent
    # For each cell (x, y), the row i for x and the column i for y must be such that:
    # if the cell is black, then row_i[x] >= y and col_i[y] >= x
    # if the cell is white, then row_i[x] < y and col_i[y] < x
    
    # We will check all constraints again
    for x, y, c in constraints:
        if x not in row_i or y not in col_i:
            # If the row or column is not constrained, it's impossible to satisfy
            print("No")
            return
        row_low, row_high = row_i[x]
        col_low, col_high = col_i[y]
        if c == 'B':
            if not (row_low <= y <= row_high and col_low <= x <= col_high):
                print("No")
                return
        else:
            if not (y > row_high or x > col_high):
                print("No")
                return
    
    # If all constraints are satisfied, print Yes
    print("Yes")

if __name__ == "__main__":
    main()