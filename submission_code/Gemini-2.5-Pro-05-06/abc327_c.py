def solve():
    A = []
    for _ in range(9):
        row = list(map(int, input().split()))
        A.append(row)

    # Check rows
    for i in range(9):
        # A[i] is the i-th row (a list of 9 numbers)
        # Check if all 9 numbers in the row are distinct.
        # Since numbers are guaranteed to be 1-9, 9 distinct numbers means they are 1-9.
        if len(set(A[i])) != 9:
            print("No")
            return

    # Check columns
    for j in range(9):  # j is the column index
        column_elements = set()
        for i in range(9):  # i is the row index
            column_elements.add(A[i][j])
        # Check if all 9 numbers in the column are distinct.
        if len(column_elements) != 9:
            print("No")
            return

    # Check 3x3 subgrids
    # Iterate over the top-left corner of each 3x3 subgrid.
    # r_start and c_start will take values 0, 3, 6.
    for r_start in range(0, 9, 3):
        for c_start in range(0, 9, 3):
            subgrid_elements = set()
            # Iterate over cells within the current 3x3 subgrid
            for i in range(r_start, r_start + 3):
                for j in range(c_start, c_start + 3):
                    subgrid_elements.add(A[i][j])
            # Check if all 9 numbers in the subgrid are distinct.
            if len(subgrid_elements) != 9:
                print("No")
                return
    
    # If all checks passed
    print("Yes")

# Call the function to execute the logic
solve()