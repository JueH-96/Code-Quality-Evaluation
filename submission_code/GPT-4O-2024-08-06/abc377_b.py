# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    # There are 8 rows and 8 columns
    n = 8
    
    # To track unsafe rows and columns
    unsafe_rows = set()
    unsafe_columns = set()
    
    # Read the grid and determine unsafe rows and columns
    for i in range(n):
        for j in range(n):
            if data[i][j] == '#':
                unsafe_rows.add(i)
                unsafe_columns.add(j)
    
    # Count the number of safe squares
    safe_count = 0
    for i in range(n):
        for j in range(n):
            if data[i][j] == '.' and i not in unsafe_rows and j not in unsafe_columns:
                safe_count += 1
    
    # Output the result
    print(safe_count)

# This ensures the main function runs when the script is executed
if __name__ == "__main__":
    main()